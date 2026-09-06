#!/usr/bin/env python3
"""Post-V5 fixed-step energy and phase convergence probe.

This additive benchmark does not change the frozen V5 solver or Challenger
contracts.  It evaluates one exactly specified, analytic equal-mass circular
binary on three fixed-step lattices.  JX KDK is always available; the optional
external lanes require the exact retained REBOUND 5.1.1 build profile.

The report is fixture-specific numerical evidence.  It does not rank engines,
establish a general convergence order, measure continuous-time invariant
envelopes, or authorize scientific/production use.
"""

from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import dataclass
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import sys
import tempfile
import time
from typing import Any, Iterable


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import numpy as np

from jxplanetx import __version__ as JX_VERSION
import jxplanetx.engine as jx_engine
from jxplanetx.engine import (
    BackendSpec,
    FIXED_STEP_KDK_METHOD_ID,
    FixedStepKDKSpec,
    ForcePlan,
    KDK_COMPOSITION,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
    integrate_kdk_trajectory,
)


SCHEMA = "jx.post-v5.energy-phase-convergence.v1"
BENCHMARK_ID = "jx.post-v5.equal-binary.fixed-step-precision.v1"
PROFILE_NAMES = ("smoke", "full-100-period")
REBOUND_MODES = ("disabled", "auto", "required")
REQUIRED_REBOUND_VERSION = "5.1.1"
EXPECTED_JX_VERSION = "0.4.0a1"
EXPECTED_NUMPY_VERSION = "2.3.5"
PACKAGE_INIT_RELATIVE_PATH = "src/jxplanetx/__init__.py"
PACKAGE_INIT_SIZE = 97
PACKAGE_INIT_SHA256 = "e1a5c7ea6928757ba8f8de69a0667dd7893d75c656123fdf42d3f430c60a9f51"
EXPECTED_NUMPY_RUNTIME = {
    "package_module": {
        "relative_path": "numpy/__init__.py",
        "size_bytes": 25_919,
        "sha256": "93924ac4b793328947dfd9eb9355e54eccdfd26a92c8dd52188aed2e52c7eb38",
    },
    "multiarray_umath_native_binary": {
        "relative_path": "numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so",
        "size_bytes": 9_270_336,
        "sha256": "ed49278029e17db9fc349d4181a2ee5a064a86bbb1241141592b8653824fd102",
    },
    "linalg_native_binary": {
        "relative_path": "numpy/linalg/_umath_linalg.cpython-314-x86_64-linux-gnu.so",
        "size_bytes": 126_592,
        "sha256": "31f694e81c9c35a59979e5cf00cf52b5724830564761bcf535f48879e0ea9a98",
    },
}
STEP_DIVISORS = (256, 512, 1024)
EXPECTED_REBOUND_ONE_STEP_ULP = {256: 0, 512: 0, 1024: 2}
SAMPLES_PER_PERIOD = 16
BODY_IDS = ("BODY_A", "BODY_B")
MASS_VALUES = (0.5, 0.5)
ENERGY_0 = -0.125
ANGULAR_MOMENTUM_0 = (0.0, 0.0, 0.25)
DECIMAL_PRECISION = 100
DECIMAL_MAXIMUM_SERIES_TERMS = 256
DECIMAL_EPOCH_ABS_MAX = 1.0e20
DECIMAL_PI_TEXT = (
    "3.141592653589793238462643383279502884197169399375105820974944592307816406"
    "2862089986280348253421170679821480865132823066470938446"
)
PERIOD = 2.0 * float(DECIMAL_PI_TEXT)

ENGINE_SOURCE_MANIFEST_SCHEMA = "jxplanetx.engine-source-manifest.v1"
ENGINE_SOURCE_TREE_HASH_DOMAIN = "jxplanetx.engine-source-tree.v1"
EXPECTED_ENGINE_SOURCE_TREE_SHA256 = (
    "974e31567db7f0e717f76944aedb0c027632f9ab2f4f9f4cbf9b66866d114269"
)
REQUIRED_ENGINE_SOURCE_PATHS = (
    "jxplanetx/engine/__init__.py",
    "jxplanetx/engine/api.py",
    "jxplanetx/engine/backends.py",
    "jxplanetx/engine/catalog.py",
    "jxplanetx/engine/contracts.py",
    "jxplanetx/engine/encounter.py",
    "jxplanetx/engine/encounter_contracts.py",
    "jxplanetx/engine/evaluator.py",
    "jxplanetx/engine/forces.py",
    "jxplanetx/engine/hybrid.py",
    "jxplanetx/engine/hybrid_contracts.py",
    "jxplanetx/engine/rkf78.py",
    "jxplanetx/engine/symplectic.py",
    "jxplanetx/engine/symplectic_contracts.py",
    "jxplanetx/engine/trajectory.py",
    "jxplanetx/engine/trajectory_contracts.py",
    "jxplanetx/engine/wisdom_holman.py",
    "jxplanetx/engine/wisdom_holman_contracts.py",
)

SUPPORT_SOURCE_RELATIVE_PATH = "benchmarks/rebound_leapfrog_comparison.py"
SUPPORT_SOURCE_SIZE = 76_043
SUPPORT_SOURCE_SHA256 = (
    "dca1f9862e4b077996398536b481ab8fc38c043159876fbe53fbec74550f34db"
)

# Exact same-build profile used by the V5 Challenger audit.  The probe remains
# usable in JX-only mode without importing REBOUND.
EXPECTED_REBOUND_PROFILE = {
    "version": "5.1.1",
    "githash": "33549d1d50d616a95a6d6a79e5e2c9c3b3730b1f",
    "module_size_bytes": 3_824,
    "module_sha256": "924e19bfddb22bca9cbff0c641eaf3b2f791287f33df2d8d852f80aac78f8cdd",
    "library_size_bytes": 2_104_024,
    "library_sha256": "52df4fe1050b98d6b0db11f31c18ec14ed00697a987e904bd7587cce4a0ccdef",
    "metadata_size_bytes": 9_389,
    "metadata_sha256": "cb8b4530f733ac4b2ed6bf385c5198f5244e2382181cf0733a08d8aff5c22bb8",
    "license_size_bytes": 35_147,
    "license_sha256": "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
    "python_source_file_count": 67,
    "python_source_tree_sha256": "e64bb210040368bfbf82d2993e8e1824f1571bd788755bc20028d41aaad3c40e",
}

EXPECTED_JX_NATIVE_DIGESTS = {
    "smoke": {
        256: {
            "schedule_content_sha256": "e95e88af7e8d0b414b1f883a884295180501c95160bc4c7e3f3bb7a49a7f7e1d",
            "result_content_sha256": "390e8a2a32121d94196a0ca0a03544c9f243bd832339ba6d3c3f300093bf86c6",
        },
        512: {
            "schedule_content_sha256": "8e0901d69a2275c4bec06b1f342a338c965b3a20991cd45f42f95be511115d2f",
            "result_content_sha256": "6d899ee4c54f07b20494ebc9ebb6842ed7b964a695f9bdb05f92cb1feb270ac3",
        },
        1024: {
            "schedule_content_sha256": "f15676fb7e11bc051e32433a1d2d629e36f24666d92db762213957240a743853",
            "result_content_sha256": "69df305a1081fff8cfffa9b4bfd8ceb5288753d754386e25711951163cea064d",
        },
    },
    "full-100-period": {
        256: {
            "schedule_content_sha256": "9658c4a03e43ebd8a481626cb8fcd1b1bb2241d6706ce3c66a427f6cbbeb3a0e",
            "result_content_sha256": "7a3549222c6a523f1868dd9f6805402691991ebe88c7778082ca3eae45898388",
        },
        512: {
            "schedule_content_sha256": "5850f7161da3879ad0b2671a67137f3c4d5bae1162677ee076d5adaf528bbd72",
            "result_content_sha256": "8d52a35ba1fa7b9dfab5e00a8ebd8659c1cf76771df2c6272d18d3052868bce7",
        },
        1024: {
            "schedule_content_sha256": "a9392483dad19c660da7c02d5317ac33b93713205033fb410be3666e3d464037",
            "result_content_sha256": "bf1f4a23edce618cf9b82a6284156c234a7d263b84f6533242838d0e82ea04ae",
        },
    },
}

EXPECTED_ANALYTIC_MATRIX_SHA256 = {
    "smoke": "b8254cb7909474701e9d71847524dac660bfff537508d1bfd91d194161cae7d6",
    "full-100-period": "aa509f562819b8af35773806c42291e74246067ea0306c4c0bee09af15a6d7b7",
}

EXPECTED_RAW_STATE_SHA256: dict[str, dict[str, str]] = {
    "smoke": {
        "jx_kdk_p256": "a68733bdc68425300fa5d2368bcfda585fc05b187cf840c10d98845c1fb1b912",
        "jx_kdk_p512": "294fe7f30dac35a6e7f364c75a5588ee9aeefa82db5e68331b3a21ca4ab78fd4",
        "jx_kdk_p1024": "41b523f54a4142154e6bb8f5ba28e11923c2e458fb37934842347a1717d188c7",
        "rebound_leapfrog_p256": "85e8be3e0a4b25f3d8f4b35e4144cbdf3a9b3e09c7cf06ad2a6776f6a072f959",
        "rebound_leapfrog_p512": "dc3d9c492ff41894085e37482ace3d68b87644a02d0a96cba70b740d313a2705",
        "rebound_leapfrog_p1024": "fed3fd2cd480540a31cb2f7b55a1ba67bdab79caa15fbb0af967a0ed80e04f66",
    },
    "full-100-period": {
        "jx_kdk_p256": "de883f9141c53479f6ac955bde7fbe8da04e605d590e5278aeff3f7b0021b6f6",
        "jx_kdk_p512": "5a3741a35c6dabcd16fb786dc188188a0264ec7632dd67ce51748c1871b627fb",
        "jx_kdk_p1024": "1986ef5ffe39da9cb245234776656a1ca1c3e47d886b8c6f829f39c366462a5f",
        "rebound_leapfrog_p256": "66b223fdc2f1ce204932ad35d20f4b3390e24313beca32f95e3e21a413246697",
        "rebound_leapfrog_p512": "72e8da348ca57ed0a7ac0310fdc54f688eea3a92b8b10aad109a3e8d2834fdda",
        "rebound_leapfrog_p1024": "ce42d48119ccb59f6ad5050fa084b3d7a25af926be7ae5cd41894f735396f198",
    },
}

MATRIX_COLUMNS = (
    "declared_epoch",
    "observed_epoch",
    "body_a_x",
    "body_a_y",
    "body_a_z",
    "body_b_x",
    "body_b_y",
    "body_b_z",
    "body_a_vx",
    "body_a_vy",
    "body_a_vz",
    "body_b_vx",
    "body_b_vy",
    "body_b_vz",
)
ANALYTIC_MATRIX_COLUMNS = (
    "declared_epoch",
    *MATRIX_COLUMNS[2:],
)

MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 300_000
MAX_JSON_STRING_BYTES = 3 * 1024 * 1024
MAX_JSON_INTEGER_BITS = 4_096
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_RUNTIME_BINARY_BYTES = 16 * 1024 * 1024
MATRIX_DOMAIN = "jx.post-v5.precision-matrix.v1"
SEMANTIC_DOMAIN = "jx.post-v5.precision-semantic.v1"
TYPED_TREE_SCHEMA = "jx.binary64-exact-typed-json-tree.v1"

CLAIM_CONTROLS = {
    "binary64_precision_convergence_claimed": False,
    "collision_encounter_or_chaos_coverage_claimed": False,
    "continuous_time_energy_envelope_claimed": False,
    "cross_hardware_bitwise_reproducibility_claimed": False,
    "equivalence_claimed": False,
    "general_or_theoretical_order_claimed": False,
    "long_term_boundedness_claimed": False,
    "production_use_authorized": False,
    "pure_python_runtime_claimed": False,
    "qualification_claimed": False,
    "reference_truth_claimed": False,
    "registry_authorized": False,
    "secular_physical_drift_claimed": False,
    "superiority_claimed": False,
    "timing_comparable": False,
}


class PrecisionProbeError(RuntimeError):
    """The closed precision-probe contract could not be satisfied."""


class ReboundUnavailable(PrecisionProbeError):
    """The exact external comparator is unavailable."""


@dataclass(frozen=True, slots=True)
class PrecisionProfile:
    name: str = "smoke"

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in PROFILE_NAMES:
            raise ValueError(f"name must be one of {PROFILE_NAMES!r}")

    @property
    def periods(self) -> int:
        return 100 if self.name == "full-100-period" else 1

    @property
    def checkpoint_count(self) -> int:
        return self.periods * SAMPLES_PER_PERIOD + 1

    def checkpoint_step_indices(self, divisor: int) -> tuple[int, ...]:
        if type(divisor) is not int or divisor not in STEP_DIVISORS:
            raise ValueError("divisor is outside the exact precision ladder")
        stride = divisor // SAMPLES_PER_PERIOD
        return tuple(index * stride for index in range(self.checkpoint_count))

    def declared_epochs(self) -> np.ndarray:
        indices = self.checkpoint_step_indices(STEP_DIVISORS[0])
        step = PERIOD / STEP_DIVISORS[0]
        return np.array(tuple(float(step * index) for index in indices), dtype="<f8")


@dataclass(frozen=True, slots=True)
class LaneRun:
    lane_id: str
    family: str
    method_id: str
    divisor: int
    declared_epochs: np.ndarray
    observed_epochs: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accounting: dict[str, Any]
    runtime: dict[str, Any]
    wall_seconds: float


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(
    path: Path,
    *,
    relative_path: str,
    maximum_bytes: int = MAX_SOURCE_BYTES,
) -> dict[str, Any]:
    if type(relative_path) is not str or not relative_path:
        raise PrecisionProbeError("source relative path is invalid")
    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= MAX_RUNTIME_BINARY_BYTES:
        raise PrecisionProbeError("file identity byte cap is invalid")
    descriptor = -1
    try:
        path_before = os.lstat(path)
        if (
            not stat.S_ISREG(path_before.st_mode)
            or path_before.st_size <= 0
            or path_before.st_size > maximum_bytes
        ):
            raise PrecisionProbeError("source is not a bounded regular file")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (path_before.st_dev, path_before.st_ino)
            or before.st_size != path_before.st_size
        ):
            raise PrecisionProbeError("source identity changed before read")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise PrecisionProbeError("source exceeds its byte cap")
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
    except OSError as exc:
        raise PrecisionProbeError("source identity could not be retained") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise PrecisionProbeError("source descriptor could not be closed") from exc

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            stat.S_IFMT(value.st_mode),
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if len(data) != before.st_size or not (
        identity(path_before) == identity(before) == identity(after) == identity(path_after)
    ):
        raise PrecisionProbeError("source changed while it was observed")
    return {
        "relative_path": relative_path,
        "size_bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _engine_source_manifest() -> dict[str, Any]:
    root = Path(jx_engine.__file__).resolve().parent
    actual_names: list[str] = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.name.endswith(".py"):
                    continue
                metadata = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                    raise PrecisionProbeError("engine source roster contains a non-regular member")
                actual_names.append(f"jxplanetx/engine/{entry.name}")
    except OSError as exc:
        raise PrecisionProbeError("engine source roster could not be inspected") from exc
    actual = tuple(sorted(actual_names))
    if actual != REQUIRED_ENGINE_SOURCE_PATHS:
        raise PrecisionProbeError("the frozen engine source roster changed")
    files: list[dict[str, Any]] = []
    for relative in actual:
        identity = _file_identity(
            root / PurePosixPath(relative).name,
            relative_path=relative,
        )
        files.append(
            {
                "path": identity["relative_path"],
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    payload = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    tree_sha256 = hashlib.sha256(
        ENGINE_SOURCE_TREE_HASH_DOMAIN.encode("ascii") + b"\0" + payload
    ).hexdigest()
    if tree_sha256 != EXPECTED_ENGINE_SOURCE_TREE_SHA256:
        raise PrecisionProbeError("the frozen engine source tree changed")
    return {
        "schema": ENGINE_SOURCE_MANIFEST_SCHEMA,
        "files": files,
        "file_count": len(files),
        "tree_hash_domain": ENGINE_SOURCE_TREE_HASH_DOMAIN,
        "tree_sha256": tree_sha256,
    }


def _support_source_identity() -> dict[str, Any]:
    identity = _file_identity(
        _REPOSITORY_ROOT / SUPPORT_SOURCE_RELATIVE_PATH,
        relative_path=SUPPORT_SOURCE_RELATIVE_PATH,
    )
    if (
        identity["size_bytes"] != SUPPORT_SOURCE_SIZE
        or identity["sha256"] != SUPPORT_SOURCE_SHA256
    ):
        raise PrecisionProbeError("the frozen analytic-comparator support source changed")
    return identity


def _package_init_identity() -> dict[str, Any]:
    identity = _file_identity(
        _REPOSITORY_ROOT / PACKAGE_INIT_RELATIVE_PATH,
        relative_path=PACKAGE_INIT_RELATIVE_PATH,
    )
    if (
        identity["size_bytes"] != PACKAGE_INIT_SIZE
        or identity["sha256"] != PACKAGE_INIT_SHA256
    ):
        raise PrecisionProbeError("the frozen package initializer changed")
    return identity


def _numpy_runtime_contract() -> dict[str, Any]:
    multiarray = importlib.import_module("numpy._core._multiarray_umath")
    linalg = importlib.import_module("numpy.linalg._umath_linalg")
    actual = {
        "version": np.__version__,
        "identity_scope": "PACKAGE_MODULE_MULTIARRAY_UMATH_AND_LINALG_NATIVE_BINARIES",
        "package_module": _file_identity(
            Path(np.__file__).resolve(),
            relative_path="numpy/__init__.py",
        ),
        "multiarray_umath_native_binary": _file_identity(
            Path(multiarray.__file__).resolve(),
            relative_path=(
                "numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so"
            ),
            maximum_bytes=MAX_RUNTIME_BINARY_BYTES,
        ),
        "linalg_native_binary": _file_identity(
            Path(linalg.__file__).resolve(),
            relative_path="numpy/linalg/_umath_linalg.cpython-314-x86_64-linux-gnu.so",
            maximum_bytes=MAX_RUNTIME_BINARY_BYTES,
        ),
        "authority_authorized": False,
    }
    expected = {
        "version": EXPECTED_NUMPY_VERSION,
        "identity_scope": "PACKAGE_MODULE_MULTIARRAY_UMATH_AND_LINALG_NATIVE_BINARIES",
        **EXPECTED_NUMPY_RUNTIME,
        "authority_authorized": False,
    }
    if not _exact_tree_equal(actual, expected):
        raise PrecisionProbeError("the exact NumPy runtime bytes changed")
    return actual


def _decimal_sin_cos(epoch: float) -> tuple[float, float]:
    if type(epoch) is not float or not math.isfinite(epoch):
        raise ValueError("epoch must be a finite built-in float")
    if abs(epoch) > DECIMAL_EPOCH_ABS_MAX:
        raise ValueError("epoch exceeds the Decimal oracle domain")
    try:
        with localcontext(
            Context(prec=DECIMAL_PRECISION, rounding=ROUND_HALF_EVEN)
        ) as context:
            value = Decimal.from_float(epoch)
            pi = Decimal(DECIMAL_PI_TEXT)
            two_pi = Decimal(2) * pi
            value %= two_pi
            if value > pi:
                value -= two_pi
            if value < -pi:
                value += two_pi
            cosine_sign = Decimal(1)
            half_pi = pi / Decimal(2)
            if value > half_pi:
                value = pi - value
                cosine_sign = Decimal(-1)
            elif value < -half_pi:
                value = -pi - value
                cosine_sign = Decimal(-1)
            squared = value * value
            sine_term = sine = value
            cosine_term = cosine = Decimal(1)
            index = 1
            while True:
                previous = (sine, cosine)
                sine_term *= -squared / Decimal((2 * index) * (2 * index + 1))
                cosine_term *= -squared / Decimal((2 * index - 1) * (2 * index))
                sine += sine_term
                cosine += cosine_term
                if previous == (sine, cosine):
                    break
                index += 1
                if index > DECIMAL_MAXIMUM_SERIES_TERMS:
                    raise ValueError("Decimal trigonometric series exceeded its term cap")
            return float(sine), float(cosine_sign * cosine)
    except InvalidOperation as exc:
        raise ValueError("Decimal argument reduction failed") from exc


def _analytic_oracle_contract() -> dict[str, Any]:
    return {
        "schema": "jx.post-v5.equal-binary-decimal-oracle.v1",
        "input_epoch": "EACH_RETAINED_BUILT_IN_BINARY64_CONVERTED_BY_DECIMAL_FROM_FLOAT",
        "decimal_precision_digits": DECIMAL_PRECISION,
        "decimal_rounding": "ROUND_HALF_EVEN",
        "decimal_adjusted_exponent_bounds": [-999_999, 999_999],
        "pi_decimal_text": DECIMAL_PI_TEXT,
        "pi_decimal_text_sha256": _sha256_bytes(DECIMAL_PI_TEXT.encode("ascii")),
        "argument_reduction": "DECIMAL_MODULO_TWO_PI_THEN_REFLECT_TO_CLOSED_MINUS_PI_OVER_2_PI_OVER_2",
        "series": "DECIMAL_SINE_AND_COSINE_TAYLOR_RECURRENCES",
        "closed_form": {
            "angular_frequency": "ONE_RADIAN_PER_TIME_UNIT",
            "body_a_position": "MINUS_ONE_HALF_TIMES_COS_T_SIN_T_ZERO",
            "body_b_position": "PLUS_ONE_HALF_TIMES_COS_T_SIN_T_ZERO",
            "body_a_velocity": "PLUS_ONE_HALF_TIMES_SIN_T_MINUS_COS_T_ZERO",
            "body_b_velocity": "MINUS_ONE_HALF_TIMES_SIN_T_PLUS_COS_T_ZERO",
            "orientation": "COUNTERCLOCKWISE_BODY_B_MINUS_BODY_A_IN_XY_AT_T_ZERO",
        },
        "termination": "BOTH_DECIMAL_SUMS_UNCHANGED_AT_FIXED_CONTEXT_PRECISION",
        "maximum_series_terms": DECIMAL_MAXIMUM_SERIES_TERMS,
        "output_rounding": "DECIMAL_TO_BUILT_IN_BINARY64_ROUND_TO_NEAREST_TIES_TO_EVEN",
        "zero_policy": "CANONICAL_POSITIVE_BINARY64_ZERO_IN_MATERIALIZED_STATE",
        "maximum_absolute_input_epoch_hex": float(DECIMAL_EPOCH_ABS_MAX).hex(),
        "directed_interval_or_correct_rounding_proof_claimed": False,
        "analytic_reference_truth_claimed": False,
    }


def analytic_trajectory(epochs: Any) -> tuple[np.ndarray, np.ndarray]:
    checked = np.asarray(epochs, dtype="<f8")
    if checked.ndim != 1 or not 1 <= checked.size <= 2_000:
        raise ValueError("epochs must be a bounded nonempty vector")
    if not np.all(np.isfinite(checked)):
        raise ValueError("epochs must be finite")
    values = tuple(_decimal_sin_cos(float(epoch)) for epoch in checked)
    sine = np.array(tuple(item[0] for item in values), dtype="<f8")[:, None]
    cosine = np.array(tuple(item[1] for item in values), dtype="<f8")[:, None]
    signs = np.array((-1.0, 1.0), dtype="<f8")[None, :]
    positions = np.zeros((checked.size, 2, 3), dtype="<f8")
    velocities = np.zeros_like(positions)
    positions[:, :, 0] = 0.5 * signs * cosine
    positions[:, :, 1] = 0.5 * signs * sine
    velocities[:, :, 0] = -0.5 * signs * sine
    velocities[:, :, 1] = 0.5 * signs * cosine
    positions[positions == 0.0] = 0.0
    velocities[velocities == 0.0] = 0.0
    return positions, velocities


def _state_and_plan(profile: PrecisionProfile) -> tuple[StateSnapshot, ForcePlan]:
    source = _support_source_identity()
    provenance = Provenance(
        "jx.post-v5.equal-binary.fixture.v1",
        SUPPORT_SOURCE_RELATIVE_PATH,
        "1",
        source["sha256"],
    )
    positions, velocities = analytic_trajectory(np.array((0.0,), dtype="<f8"))
    masses = np.array(MASS_VALUES, dtype="<f8")
    snapshot = StateSnapshot(
        "jx.post-v5.equal-binary.snapshot.v1",
        0.0,
        "SYNTHETIC",
        "BARYCENTRIC_INERTIAL",
        "BARYCENTER",
        "CARTESIAN_RIGHT_HANDED",
        "L",
        "T",
        "M",
        "jx.post-v5.equal-binary.units.v1",
        BODY_IDS,
        positions[0].copy(),
        velocities[0].copy(),
        masses.copy(),
        masses.copy(),
        np.zeros(2, dtype="<f8"),
        np.ones(2, dtype=np.bool_),
        provenance,
    )
    metadata = ParameterMetadata(
        "state.gravitational_parameters",
        "L^3/T^2",
        provenance,
        None,
        None,
        0.0,
        float(profile.declared_epochs()[-1]),
    )
    plan = ForcePlan(
        "jx.post-v5.equal-binary.newtonian-plan.v1",
        BackendSpec("numpy", "cpu", 2),
        (
            NewtonianPointMass(
                BODY_IDS,
                BODY_IDS,
                "jx.post-v5.equal-binary.units.v1",
                (metadata,),
            ),
        ),
    )
    return snapshot, plan


def _readonly(array: Any, shape: tuple[int, ...]) -> np.ndarray:
    value = np.array(array, dtype="<f8", order="C", copy=True).reshape(shape).copy(order="C")
    if not np.all(np.isfinite(value)):
        raise PrecisionProbeError("lane returned nonfinite state")
    value.setflags(write=False)
    return value


def _expected_jx_effective_settings(
    profile: PrecisionProfile, divisor: int
) -> dict[str, Any]:
    return {
        "method_id": FIXED_STEP_KDK_METHOD_ID,
        "principal_order": 2,
        "composition": KDK_COMPOSITION,
        "fixed_step": PERIOD / divisor,
        "fixed_step_hex": float(PERIOD / divisor).hex(),
        "checkpoint_integer_stride": divisor // SAMPLES_PER_PERIOD,
        "checkpoint_count": profile.checkpoint_count,
        "final_integer_step": profile.periods * divisor,
        "minimum_swept_pair_separation": 0.1,
        "maximum_pair_frequency_step": 0.05,
        "dense_output": False,
        "adaptive": False,
        "backend": {
            "backend_id": "numpy",
            "device": "cpu",
            "tile_size": 2,
            "dtype": "float64",
            "allow_fallback": False,
            "deterministic_reductions": True,
            "fast_math": False,
            "determinism_scope": "SAME_RUNTIME_DEVICE",
        },
        "force_models": [
            {
                "model_id": "force.newtonian.point_mass",
                "source_ids": list(BODY_IDS),
                "target_ids": list(BODY_IDS),
                "unit_system_id": "jx.post-v5.equal-binary.units.v1",
                "softening": "NONE",
            }
        ],
        "effective_readback_bound": True,
        "qualification_authorized": False,
    }


def _jx_effective_settings(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: FixedStepKDKSpec,
    profile: PrecisionProfile,
    divisor: int,
) -> dict[str, Any]:
    backend = plan.backend
    model = plan.models[0]
    result = {
        "method_id": spec.method_id,
        "principal_order": spec.principal_order,
        "composition": spec.composition,
        "fixed_step": spec.fixed_step,
        "fixed_step_hex": spec.fixed_step.hex(),
        "checkpoint_integer_stride": (
            spec.checkpoint_step_indices[1] - spec.checkpoint_step_indices[0]
        ),
        "checkpoint_count": len(spec.checkpoint_step_indices),
        "final_integer_step": spec.checkpoint_step_indices[-1],
        "minimum_swept_pair_separation": spec.minimum_swept_pair_separation,
        "maximum_pair_frequency_step": spec.maximum_pair_frequency_step,
        "dense_output": spec.dense_output,
        "adaptive": spec.adaptive,
        "backend": {
            "backend_id": backend.backend_id,
            "device": backend.device,
            "tile_size": backend.tile_size,
            "dtype": backend.dtype,
            "allow_fallback": backend.allow_fallback,
            "deterministic_reductions": backend.deterministic_reductions,
            "fast_math": backend.fast_math,
            "determinism_scope": backend.determinism_scope,
        },
        "force_models": [
            {
                "model_id": model.model_id,
                "source_ids": list(model.source_ids),
                "target_ids": list(model.target_ids),
                "unit_system_id": model.unit_system_id,
                "softening": "NONE",
            }
        ],
        "effective_readback_bound": True,
        "qualification_authorized": plan.qualification_authorized,
    }
    if snapshot.body_ids != BODY_IDS or not _exact_tree_equal(
        result, _expected_jx_effective_settings(profile, divisor)
    ):
        raise PrecisionProbeError("JX effective settings differ from the exact profile")
    return result


def _run_jx_lane(profile: PrecisionProfile, divisor: int) -> LaneRun:
    if JX_VERSION != EXPECTED_JX_VERSION or np.__version__ != EXPECTED_NUMPY_VERSION:
        raise PrecisionProbeError("the exact JX/NumPy precision runtime changed")
    declared = profile.declared_epochs()
    indices = profile.checkpoint_step_indices(divisor)
    snapshot, plan = _state_and_plan(profile)
    spec = FixedStepKDKSpec(
        checkpoint_step_indices=indices,
        fixed_step=PERIOD / divisor,
        maximum_steps=indices[-1],
        minimum_swept_pair_separation=0.1,
        maximum_pair_frequency_step=0.05,
    )
    effective_settings = _jx_effective_settings(
        snapshot, plan, spec, profile, divisor
    )
    started = time.perf_counter()
    result = integrate_kdk_trajectory(snapshot, plan, spec)
    elapsed = time.perf_counter() - started
    observed = _readonly(result.checkpoint_epochs, (profile.checkpoint_count,))
    positions = _readonly(np.stack(result.positions), (profile.checkpoint_count, 2, 3))
    velocities = _readonly(np.stack(result.velocities), (profile.checkpoint_count, 2, 3))
    if not _exact_array_equal(observed, declared):
        raise PrecisionProbeError("JX checkpoint clock differs from the shared lattice")
    accounting = {
        "checkpoint_count": result.checkpoint_count,
        "completed_steps": result.completed_steps,
        "force_evaluations": result.force_evaluations,
        "schedule_content_sha256": result.schedule_content_sha256,
        "result_content_sha256": result.result_content_sha256,
        "engine_internal_trajectory_reexecution_count": 0,
        "engine_result_constructor_validation_performed": True,
        "effective_settings": effective_settings,
    }
    expected_steps = profile.periods * divisor
    if accounting != {
        "checkpoint_count": profile.checkpoint_count,
        "completed_steps": expected_steps,
        "force_evaluations": expected_steps + 1,
        "schedule_content_sha256": result.schedule_content_sha256,
        "result_content_sha256": result.result_content_sha256,
        "engine_internal_trajectory_reexecution_count": 0,
        "engine_result_constructor_validation_performed": True,
        "effective_settings": effective_settings,
    } or not all(
        _is_sha256(accounting[field])
        for field in ("schedule_content_sha256", "result_content_sha256")
    ):
        raise PrecisionProbeError("JX work accounting changed")
    expected_native = EXPECTED_JX_NATIVE_DIGESTS[profile.name][divisor]
    if any(accounting[key] != value for key, value in expected_native.items()):
        raise PrecisionProbeError("JX native schedule or result digest changed")
    return LaneRun(
        f"jx_kdk_p{divisor}",
        "jx_kdk",
        FIXED_STEP_KDK_METHOD_ID,
        divisor,
        _readonly(declared, declared.shape),
        observed,
        positions,
        velocities,
        accounting,
        _jx_runtime_contract(),
        float(elapsed),
    )


def _jx_runtime_contract() -> dict[str, Any]:
    return {
        "jxplanetx_version": EXPECTED_JX_VERSION,
        "engine_source_tree_sha256": EXPECTED_ENGINE_SOURCE_TREE_SHA256,
        "numpy": _numpy_runtime_contract(),
    }


def _distribution_file(distribution: Any, suffix: str) -> Path:
    matches = [
        entry
        for entry in distribution.files or ()
        if str(entry).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ReboundUnavailable(f"REBOUND distribution lacks one exact {suffix}")
    return Path(os.path.abspath(os.fspath(distribution.locate_file(matches[0]))))


def _rebound_python_source_tree(distribution: Any) -> tuple[int, str]:
    entries: list[dict[str, Any]] = []
    for entry in distribution.files or ():
        relative = str(entry)
        path = PurePosixPath(relative)
        if (
            relative.startswith("/")
            or "\\" in relative
            or ".." in path.parts
            or path.as_posix() != relative
        ):
            raise ReboundUnavailable("REBOUND distribution path is not normalized")
        if relative.startswith("rebound/") and relative.endswith(".py"):
            identity = _file_identity(
                Path(os.path.abspath(os.fspath(distribution.locate_file(entry)))),
                relative_path=relative,
            )
            entries.append(identity)
    entries.sort(key=lambda item: item["relative_path"])
    payload = json.dumps(
        [
            {
                "path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in entries
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256(
        b"rebound.distribution-python-source-tree.v1\0" + payload
    ).hexdigest()
    return len(entries), digest


def _load_exact_rebound() -> tuple[Any, dict[str, Any]]:
    try:
        rebound = importlib.import_module("rebound")
        distribution = importlib.metadata.distribution("rebound")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ReboundUnavailable("REBOUND 5.1.1 is not installed") from exc
    if rebound.__version__ != REQUIRED_REBOUND_VERSION or distribution.version != REQUIRED_REBOUND_VERSION:
        raise PrecisionProbeError("installed REBOUND is not exactly 5.1.1")
    module_path = Path(rebound.__file__).resolve()
    library_path = Path(rebound.clibrebound._name).resolve()
    metadata_path = _distribution_file(distribution, ".dist-info/METADATA")
    license_path = _distribution_file(distribution, ".dist-info/licenses/LICENSE")
    module = _file_identity(module_path, relative_path="rebound/__init__.py")
    library = _file_identity(library_path, relative_path=library_path.name)
    metadata = _file_identity(metadata_path, relative_path="rebound-5.1.1.dist-info/METADATA")
    license_file = _file_identity(
        license_path,
        relative_path="rebound-5.1.1.dist-info/licenses/LICENSE",
    )
    source_count, source_tree = _rebound_python_source_tree(distribution)
    runtime = {
        "version": rebound.__version__,
        "githash": getattr(rebound, "__githash__", None),
        "module": module,
        "library": library,
        "metadata": metadata,
        "license": license_file,
        "python_source_file_count": source_count,
        "python_source_tree_sha256": source_tree,
        "license_expression": distribution.metadata.get("License-Expression"),
        "authority_authorized": False,
    }
    expected = EXPECTED_REBOUND_PROFILE
    if (
        runtime["version"] != expected["version"]
        or runtime["githash"] != expected["githash"]
        or module["size_bytes"] != expected["module_size_bytes"]
        or module["sha256"] != expected["module_sha256"]
        or library["size_bytes"] != expected["library_size_bytes"]
        or library["sha256"] != expected["library_sha256"]
        or metadata["size_bytes"] != expected["metadata_size_bytes"]
        or metadata["sha256"] != expected["metadata_sha256"]
        or license_file["size_bytes"] != expected["license_size_bytes"]
        or license_file["sha256"] != expected["license_sha256"]
        or source_count != expected["python_source_file_count"]
        or source_tree != expected["python_source_tree_sha256"]
        or runtime["license_expression"] != "GPL-3.0-only"
    ):
        raise PrecisionProbeError("REBOUND 5.1.1 bytes differ from the exact V5 profile")
    return rebound, runtime


def _configured_rebound_simulation(rebound: Any, divisor: int) -> Any:
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    simulation.integrator = "leapfrog"
    simulation.dt = PERIOD / divisor
    simulation.add(m=0.5, r=0.0, x=-0.5, y=0.0, z=0.0, vx=0.0, vy=-0.5, vz=0.0)
    simulation.add(m=0.5, r=0.0, x=0.5, y=0.0, z=0.0, vx=0.0, vy=0.5, vz=0.0)
    simulation.N_active = 2
    simulation.testparticle_type = 0
    if not callable(getattr(simulation, "steps", None)):
        raise PrecisionProbeError("REBOUND does not expose Simulation.steps")
    return simulation


def _expected_rebound_effective_settings(divisor: int) -> dict[str, Any]:
    if type(divisor) is not int or divisor not in STEP_DIVISORS:
        raise PrecisionProbeError("REBOUND settings divisor changed")
    return {
        "G": 1.0,
        "gravity": "basic",
        "collision": "none",
        "boundary": "none",
        "softening": 0.0,
        "integrator": "leapfrog",
        "integrator_order": 2,
        "method_variant": "REBOUND_LEAPFROG_DRIFT_KICK_DRIFT",
        "equivalent_to_jx_kdk_map_claimed": False,
        "dt": PERIOD / divisor,
        "dt_hex": float(PERIOD / divisor).hex(),
        "N": 2,
        "N_active": 2,
        "testparticle_type": 0,
        "initial_particle_effective_readback": [
            {
                "m": 0.5,
                "r": 0.0,
                "x": -0.5,
                "y": 0.0,
                "z": 0.0,
                "vx": 0.0,
                "vy": -0.5,
                "vz": 0.0,
            },
            {
                "m": 0.5,
                "r": 0.0,
                "x": 0.5,
                "y": 0.0,
                "z": 0.0,
                "vx": 0.0,
                "vy": 0.5,
                "vz": 0.0,
            },
        ],
        "advance_api": "Simulation.steps(integer_count)",
        "effective_readback_bound": True,
        "qualification_authorized": False,
    }


def _rebound_effective_settings(simulation: Any, divisor: int) -> dict[str, Any]:
    integrator = simulation.integrator
    result = {
        "G": float(simulation.G),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening": float(simulation.softening),
        "integrator": str(integrator),
        "integrator_order": int(integrator.order),
        "method_variant": "REBOUND_LEAPFROG_DRIFT_KICK_DRIFT",
        "equivalent_to_jx_kdk_map_claimed": False,
        "dt": float(simulation.dt),
        "dt_hex": float(simulation.dt).hex(),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
        "initial_particle_effective_readback": [
            {
                field: float(getattr(particle, field))
                for field in ("m", "r", "x", "y", "z", "vx", "vy", "vz")
            }
            for particle in simulation.particles[:2]
        ],
        "advance_api": "Simulation.steps(integer_count)",
        "effective_readback_bound": True,
        "qualification_authorized": False,
    }
    if _typed_tree(result) != _typed_tree(_expected_rebound_effective_settings(divisor)):
        raise PrecisionProbeError("REBOUND effective settings differ from the exact profile")
    return result


def _one_step_dkd_evidence(rebound: Any, divisor: int) -> dict[str, Any]:
    simulation = _configured_rebound_simulation(rebound, divisor)
    step = PERIOD / divisor
    positions = np.array(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype="<f8")
    velocities = np.array(((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)), dtype="<f8")
    half = positions + (0.5 * step) * velocities
    displacement = half[1] - half[0]
    squared = float(np.dot(displacement, displacement))
    inverse_cube = 1.0 / (squared * math.sqrt(squared))
    acceleration = np.empty_like(positions)
    acceleration[0] = 0.5 * displacement * inverse_cube
    acceleration[1] = -0.5 * displacement * inverse_cube
    expected_velocity = velocities + step * acceleration
    expected_position = half + (0.5 * step) * expected_velocity
    simulation.steps(1)
    observed_position = np.array(
        tuple((p.x, p.y, p.z) for p in simulation.particles[:2]), dtype="<f8"
    )
    observed_velocity = np.array(
        tuple((p.vx, p.vy, p.vz) for p in simulation.particles[:2]), dtype="<f8"
    )

    def ordered_ulp_distance(left: float, right: float) -> int:
        if not math.isfinite(left) or not math.isfinite(right):
            raise PrecisionProbeError("one-step DKD comparison contains nonfinite state")
        first_bits = int(np.array((left,), dtype="<f8").view("<u8")[0])
        second_bits = int(np.array((right,), dtype="<f8").view("<u8")[0])
        if left == 0.0 or right == 0.0:
            if first_bits != second_bits:
                raise PrecisionProbeError("one-step DKD comparison changed a signed-zero bit")
            return 0

        def ordered(bits: int) -> int:
            return (
                (~bits) & 0xFFFFFFFFFFFFFFFF
                if bits & 0x8000000000000000
                else bits | 0x8000000000000000
            )

        return abs(ordered(first_bits) - ordered(second_bits))

    maximum_ulp = max(
        ordered_ulp_distance(float(left), float(right))
        for left, right in zip(
            np.concatenate((expected_position.ravel(), expected_velocity.ravel())),
            np.concatenate((observed_position.ravel(), observed_velocity.ravel())),
            strict=True,
        )
    )
    if maximum_ulp > 4:
        raise PrecisionProbeError("REBOUND one-step DKD evidence exceeded four ULP")
    if maximum_ulp != EXPECTED_REBOUND_ONE_STEP_ULP[divisor]:
        raise PrecisionProbeError("REBOUND one-step DKD same-build regression changed")
    return {
        "equation": "BENCHMARK_LOCAL_BINARY64_DRIFT_KICK_DRIFT",
        "maximum_component_ulp_distance": maximum_ulp,
        "maximum_allowed_component_ulp_distance": 4,
        "zero_and_signed_zero_policy": "EXACT_BINARY64_BITS",
        "bitwise_same_runtime_replay_required_separately": True,
        "qualification_authorized": False,
    }


def _run_rebound_lane(
    profile: PrecisionProfile,
    divisor: int,
    rebound: Any,
    runtime: dict[str, Any],
) -> LaneRun:
    simulation = _configured_rebound_simulation(rebound, divisor)
    effective_settings = _rebound_effective_settings(simulation, divisor)
    evidence = _one_step_dkd_evidence(rebound, divisor)
    declared = profile.declared_epochs()
    indices = profile.checkpoint_step_indices(divisor)
    observed = np.empty(profile.checkpoint_count, dtype="<f8")
    positions = np.empty((profile.checkpoint_count, 2, 3), dtype="<f8")
    velocities = np.empty_like(positions)

    def capture(index: int) -> None:
        observed[index] = float(simulation.t)
        for body, particle in enumerate(simulation.particles[:2]):
            positions[index, body] = (particle.x, particle.y, particle.z)
            velocities[index, body] = (particle.vx, particle.vy, particle.vz)

    capture(0)
    arguments: list[int] = []
    previous = 0
    started = time.perf_counter()
    for output_index, current in enumerate(indices[1:], 1):
        count = current - previous
        if type(count) is not int or count <= 0:
            raise PrecisionProbeError("REBOUND step-call lattice changed")
        simulation.steps(count)
        arguments.append(count)
        capture(output_index)
        previous = current
    elapsed = time.perf_counter() - started
    expected_steps = profile.periods * divisor
    if sum(arguments) != expected_steps or len(arguments) != profile.checkpoint_count - 1:
        raise PrecisionProbeError("REBOUND step accounting changed")
    return LaneRun(
        f"rebound_leapfrog_p{divisor}",
        "rebound_leapfrog",
        "rebound.integrator.leapfrog.5.1.1",
        divisor,
        _readonly(declared, declared.shape),
        _readonly(observed, observed.shape),
        _readonly(positions, positions.shape),
        _readonly(velocities, velocities.shape),
        {
            "checkpoint_count": profile.checkpoint_count,
            "completed_integer_steps": expected_steps,
            "simulation_steps_call_count": len(arguments),
            "simulation_steps_argument": divisor // SAMPLES_PER_PERIOD,
            "force_evaluations": None,
            "one_step_dkd_evidence": evidence,
            "effective_settings": effective_settings,
        },
        copy.deepcopy(runtime),
        float(elapsed),
    )


def _matrix_from_lane(lane: LaneRun) -> np.ndarray:
    return np.ascontiguousarray(
        np.column_stack(
            (
                lane.declared_epochs,
                lane.observed_epochs,
                lane.positions.reshape((lane.positions.shape[0], 6)),
                lane.velocities.reshape((lane.velocities.shape[0], 6)),
            )
        ),
        dtype="<f8",
    )


def _analytic_matrix(epochs: np.ndarray) -> np.ndarray:
    positions, velocities = analytic_trajectory(epochs)
    return np.ascontiguousarray(
        np.column_stack(
            (epochs, positions.reshape((epochs.size, 6)), velocities.reshape((epochs.size, 6)))
        ),
        dtype="<f8",
    )


def _matrix_digest(raw: bytes, shape: tuple[int, int], columns: tuple[str, ...]) -> str:
    metadata = json.dumps(
        {"columns": list(columns), "dtype": "<f8", "shape": list(shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(MATRIX_DOMAIN.encode("ascii") + b"\0" + metadata + b"\0" + raw).hexdigest()


def _matrix_evidence(matrix: np.ndarray, columns: tuple[str, ...]) -> dict[str, Any]:
    checked = np.ascontiguousarray(matrix, dtype="<f8")
    if checked.ndim != 2 or checked.shape[1] != len(columns) or not np.all(np.isfinite(checked)):
        raise PrecisionProbeError("matrix evidence has invalid shape or values")
    raw = checked.tobytes(order="C")
    return {
        "encoding": "RFC4648_BASE64_OF_C_ORDER_LITTLE_ENDIAN_BINARY64",
        "dtype": "<f8",
        "shape": list(checked.shape),
        "columns": list(columns),
        "byte_length": len(raw),
        "sha256_domain": MATRIX_DOMAIN,
        "sha256": _matrix_digest(raw, checked.shape, columns),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }


def _profile_name_for_matrix_rows(rows: int) -> str:
    if type(rows) is not int:
        raise PrecisionProbeError("matrix row count must be an exact integer")
    matches = [
        name for name in PROFILE_NAMES if PrecisionProfile(name).checkpoint_count == rows
    ]
    if len(matches) != 1:
        raise PrecisionProbeError("matrix row count does not identify one exact profile")
    return matches[0]


def _decode_matrix_evidence(
    evidence: Any,
    *,
    expected_rows: int,
    expected_columns: tuple[str, ...],
) -> np.ndarray:
    if type(evidence) is not dict or set(evidence) != {
        "encoding", "dtype", "shape", "columns", "byte_length", "sha256_domain", "sha256", "data_base64"
    }:
        raise PrecisionProbeError("matrix evidence schema changed")
    expected_shape = [expected_rows, len(expected_columns)]
    expected_metadata = {
        "encoding": "RFC4648_BASE64_OF_C_ORDER_LITTLE_ENDIAN_BINARY64",
        "dtype": "<f8",
        "shape": expected_shape,
        "columns": list(expected_columns),
        "byte_length": expected_rows * len(expected_columns) * 8,
        "sha256_domain": MATRIX_DOMAIN,
    }
    actual_metadata = {
        key: evidence[key]
        for key in expected_metadata
    }
    if (
        not _exact_tree_equal(actual_metadata, expected_metadata)
        or not _is_sha256(evidence["sha256"])
        or type(evidence["data_base64"]) is not str
    ):
        raise PrecisionProbeError("matrix evidence metadata changed")
    try:
        raw = base64.b64decode(evidence["data_base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise PrecisionProbeError("matrix base64 is invalid") from exc
    if (
        len(raw) != evidence["byte_length"]
        or base64.b64encode(raw).decode("ascii") != evidence["data_base64"]
        or _matrix_digest(raw, tuple(expected_shape), expected_columns) != evidence["sha256"]
    ):
        raise PrecisionProbeError("matrix evidence bytes changed")
    result = np.frombuffer(raw, dtype="<f8").reshape(tuple(expected_shape)).copy(order="C")
    if not np.all(np.isfinite(result)):
        raise PrecisionProbeError("matrix contains nonfinite values")
    return result


def _scalar_summary(values: Iterable[float], abscissa: Iterable[float]) -> dict[str, float]:
    series = tuple(float(value) for value in values)
    x_values = tuple(float(value) for value in abscissa)
    if not series or len(series) != len(x_values) or not all(math.isfinite(v) for v in (*series, *x_values)):
        raise PrecisionProbeError("metric series is invalid")
    mean_x = math.fsum(x_values) / len(x_values)
    mean_y = math.fsum(series) / len(series)
    denominator = math.fsum((value - mean_x) ** 2 for value in x_values)
    slope = (
        math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, series, strict=True))
        / denominator
        if denominator > 0.0
        else 0.0
    )
    return {
        "maximum_abs": max(abs(value) for value in series),
        "rms": math.sqrt(math.fsum(value * value for value in series) / len(series)),
        "final_signed": series[-1],
        "minimum_signed": min(series),
        "maximum_signed": max(series),
        "peak_to_peak": max(series) - min(series),
        "ols_slope_per_orbit": slope,
    }


def _nonnegative_summary(values: Iterable[float]) -> dict[str, float]:
    series = tuple(float(value) for value in values)
    if not series or not all(math.isfinite(value) and value >= 0.0 for value in series):
        raise PrecisionProbeError("nonnegative metric series is invalid")
    return {
        "maximum": max(series),
        "rms": math.sqrt(math.fsum(value * value for value in series) / len(series)),
        "final": series[-1],
        "minimum": min(series),
    }


def _cartesian_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.shape != reference.shape or candidate.ndim != 3 or candidate.shape[1:] != (2, 3):
        raise PrecisionProbeError("Cartesian metric arrays have wrong shape")
    difference = candidate - reference
    if not np.all(np.isfinite(difference)):
        raise PrecisionProbeError("Cartesian difference is nonfinite")
    vector = np.linalg.norm(difference, axis=2)
    return {
        "component_maximum_abs": float(np.max(np.abs(difference))),
        "component_rms": float(np.sqrt(np.mean(difference * difference))),
        "vector_l2_maximum": float(np.max(vector)),
        "vector_l2_rms": float(np.sqrt(np.mean(vector * vector))),
    }


def _phase_metrics(candidate: np.ndarray, reference: np.ndarray, orbit_axis: np.ndarray) -> dict[str, Any]:
    candidate_relative = candidate[:, 1, :2] - candidate[:, 0, :2]
    reference_relative = reference[:, 1, :2] - reference[:, 0, :2]
    if (
        candidate_relative.shape != reference_relative.shape
        or candidate_relative.shape[0] != orbit_axis.size
        or np.any(np.linalg.norm(candidate_relative, axis=1) <= 0.0)
        or np.any(np.linalg.norm(reference_relative, axis=1) <= 0.0)
    ):
        raise PrecisionProbeError("phase vectors are invalid")
    cross = reference_relative[:, 0] * candidate_relative[:, 1]
    cross -= reference_relative[:, 1] * candidate_relative[:, 0]
    dot = np.sum(reference_relative * candidate_relative, axis=1)
    wrapped = np.arctan2(cross, dot)
    unwrapped = np.unwrap(wrapped)
    maximum_increment = float(np.max(np.abs(np.diff(wrapped)))) if wrapped.size > 1 else 0.0
    if maximum_increment >= math.pi / 2.0:
        raise PrecisionProbeError("phase sampling is ambiguous for the declared unwrap gate")
    summary = _scalar_summary(unwrapped, orbit_axis)
    summary.update(
        {
            "observable": "RELATIVE_BODY_B_MINUS_A_XY_ANGLE",
            "sign_convention": "CANDIDATE_MINUS_DECIMAL_ANALYTIC",
            "unwrap_policy": "NUMPY_UNWRAP_WITH_MAXIMUM_RAW_INCREMENT_STRICTLY_BELOW_PI_OVER_2",
            "maximum_abs_raw_increment": maximum_increment,
        }
    )
    return summary


def _invariant_metrics(positions: np.ndarray, velocities: np.ndarray, orbit_axis: np.ndarray) -> dict[str, Any]:
    separation = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1)
    if np.any(separation <= 0.0):
        raise PrecisionProbeError("binary separation is nonpositive")
    masses = np.array(MASS_VALUES, dtype="<f8")
    energy = 0.5 * np.sum(masses[None, :, None] * velocities * velocities, axis=(1, 2))
    energy -= masses[0] * masses[1] / separation
    energy_relative = (energy - ENERGY_0) / abs(ENERGY_0)
    angular = np.sum(masses[None, :, None] * np.cross(positions, velocities), axis=1)
    angular_reference = np.array(ANGULAR_MOMENTUM_0, dtype="<f8")
    angular_relative = np.linalg.norm(angular - angular_reference, axis=1) / np.linalg.norm(angular_reference)
    angular_z_signed = (angular[:, 2] - angular_reference[2]) / abs(angular_reference[2])
    center = np.sum(masses[None, :, None] * positions, axis=1) / np.sum(masses)
    momentum = np.sum(masses[None, :, None] * velocities, axis=1)
    return {
        "relative_total_energy": _scalar_summary(energy_relative, orbit_axis),
        "relative_angular_momentum_vector_norm": _nonnegative_summary(angular_relative),
        "relative_angular_momentum_z_signed": _scalar_summary(angular_z_signed, orbit_axis),
        "center_of_mass_position_norm": _nonnegative_summary(np.linalg.norm(center, axis=1)),
        "total_momentum_norm": _nonnegative_summary(np.linalg.norm(momentum, axis=1)),
        "sampling_scope": "RETAINED_OUTPUT_CHECKPOINTS_ONLY",
        "energy_baseline": ENERGY_0,
        "relative_energy_definition": "SIGNED_E_MINUS_E0_DIVIDED_BY_ABS_E0",
        "angular_momentum_baseline": list(ANGULAR_MOMENTUM_0),
        "relative_angular_momentum_z_definition": "SIGNED_LZ_MINUS_LZ0_DIVIDED_BY_ABS_LZ0",
        "invariant_ols_abscissa": "LANE_OBSERVED_CLOCK_ELAPSED_ORBITS",
    }


def _clock_metrics(declared: np.ndarray, observed: np.ndarray, orbit_axis: np.ndarray) -> dict[str, Any]:
    difference = observed - declared
    result = _scalar_summary(difference, orbit_axis)
    result["observed_clock_source"] = "LANE_EFFECTIVE_READBACK"
    return result


def _lane_metrics(matrix: np.ndarray) -> dict[str, Any]:
    declared = matrix[:, 0]
    observed = matrix[:, 1]
    positions = matrix[:, 2:8].reshape((-1, 2, 3))
    velocities = matrix[:, 8:14].reshape((-1, 2, 3))
    observed_positions, observed_velocities = analytic_trajectory(observed)
    declared_positions, declared_velocities = analytic_trajectory(declared)
    orbit_axis = (observed - observed[0]) / PERIOD
    declared_orbit_axis = (declared - declared[0]) / PERIOD
    return {
        "observed_clock_analytic_accuracy": {
            "position": _cartesian_metrics(positions, observed_positions),
            "velocity": _cartesian_metrics(velocities, observed_velocities),
            "phase": _phase_metrics(positions, observed_positions, orbit_axis),
        },
        "declared_grid_analytic_accuracy": {
            "position": _cartesian_metrics(positions, declared_positions),
            "velocity": _cartesian_metrics(velocities, declared_velocities),
            "phase": _phase_metrics(positions, declared_positions, declared_orbit_axis),
        },
        "invariants": _invariant_metrics(positions, velocities, orbit_axis),
        "clock_error": _clock_metrics(declared, observed, declared_orbit_axis),
    }


def _lane_summary(lane: LaneRun, replay: LaneRun) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        lane.lane_id != replay.lane_id
        or lane.family != replay.family
        or lane.method_id != replay.method_id
        or lane.divisor != replay.divisor
        or not _exact_tree_equal(lane.accounting, replay.accounting)
        or not _exact_tree_equal(lane.runtime, replay.runtime)
    ):
        raise PrecisionProbeError("lane replay metadata changed")
    matrix = _matrix_from_lane(lane)
    replay_matrix = _matrix_from_lane(replay)
    evidence = _matrix_evidence(matrix, MATRIX_COLUMNS)
    replay_evidence = _matrix_evidence(replay_matrix, MATRIX_COLUMNS)
    profile_name = _profile_name_for_matrix_rows(matrix.shape[0])
    expected_raw = EXPECTED_RAW_STATE_SHA256[profile_name].get(lane.lane_id)
    if expected_raw is None or evidence["sha256"] != expected_raw:
        raise PrecisionProbeError("lane raw-state regression KAT changed")
    if evidence["sha256"] != replay_evidence["sha256"] or not _exact_array_equal(matrix, replay_matrix):
        raise PrecisionProbeError("lane replay state changed")
    summary = {
        "lane_id": lane.lane_id,
        "family": lane.family,
        "role": "JX_FIXED_GRID" if lane.family == "jx_kdk" else "OPTIONAL_EXTERNAL_COMPARATOR",
        "method_id": lane.method_id,
        "divisor": lane.divisor,
        "fixed_step_hex": float(PERIOD / lane.divisor).hex(),
        "raw_state_evidence": evidence,
        "metrics": _lane_metrics(matrix),
        "accounting": copy.deepcopy(lane.accounting),
        "runtime": copy.deepcopy(lane.runtime),
        "replay": {
            "scope": "SAME_RUNTIME_COMPLETE_FRESH_STATE_REEXECUTION",
            "raw_state_sha256": replay_evidence["sha256"],
            "status": "BITWISE_IDENTICAL",
            "process_independence_claimed": False,
        },
    }
    diagnostics = {
        "lane_id": lane.lane_id,
        "primary_wall_seconds": lane.wall_seconds,
        "replay_wall_seconds": replay.wall_seconds,
    }
    return summary, diagnostics


def _metric_value(lane: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = lane
    for key in path:
        value = value[key]
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise PrecisionProbeError("convergence metric must be finite and positive")
    return value


def _convergence_summary(
    lanes: list[dict[str, Any]],
    profile: PrecisionProfile,
    *,
    accuracy_basis: str,
) -> dict[str, Any]:
    if [lane["divisor"] for lane in lanes] != list(STEP_DIVISORS):
        raise PrecisionProbeError("convergence lane order changed")
    if accuracy_basis not in (
        "observed_clock_analytic_accuracy",
        "declared_grid_analytic_accuracy",
    ):
        raise PrecisionProbeError("convergence accuracy basis is invalid")
    paths = {
        "position_vector_l2_maximum": (
            "metrics", accuracy_basis, "position", "vector_l2_maximum"
        ),
        "position_vector_l2_rms": (
            "metrics", accuracy_basis, "position", "vector_l2_rms"
        ),
        "velocity_vector_l2_maximum": (
            "metrics", accuracy_basis, "velocity", "vector_l2_maximum"
        ),
        "velocity_vector_l2_rms": (
            "metrics", accuracy_basis, "velocity", "vector_l2_rms"
        ),
        "phase_maximum_abs": (
            "metrics", accuracy_basis, "phase", "maximum_abs"
        ),
        "phase_rms": (
            "metrics", accuracy_basis, "phase", "rms"
        ),
    }
    metric_records: dict[str, Any] = {}
    for label, path in paths.items():
        values = [_metric_value(lane, path) for lane in lanes]
        orders = [math.log2(values[index] / values[index + 1]) for index in (0, 1)]
        monotonic = values[0] > values[1] > values[2]
        if not monotonic:
            raise PrecisionProbeError(f"{label} did not refine monotonically")
        status = all(1.8 <= order <= 2.2 for order in orders)
        interpretation = "FIXTURE_SPECIFIC_STATE_OR_PHASE_EMPIRICAL_ORDER"
        if not status:
            raise PrecisionProbeError(f"{label} empirical order is outside its gate")
        metric_records[label] = {
            "values_p256_p512_p1024": values,
            "adjacent_orders": orders,
            "strictly_monotonic": True,
            "interpretation": interpretation,
        }
    fine = lanes[-1]["metrics"]
    thresholds = {
        "position_vector_l2_maximum": 5.0e-3,
        "velocity_vector_l2_maximum": 5.0e-3,
        "phase_maximum_abs": 1.0e-2,
    }
    observed = {
        "position_vector_l2_maximum": fine[accuracy_basis]["position"]["vector_l2_maximum"],
        "velocity_vector_l2_maximum": fine[accuracy_basis]["velocity"]["vector_l2_maximum"],
        "phase_maximum_abs": fine[accuracy_basis]["phase"]["maximum_abs"],
    }
    if profile.name == "full-100-period" and any(
        observed[key] > threshold for key, threshold in thresholds.items()
    ):
        raise PrecisionProbeError("the finest full-profile lane exceeded an absolute gate")
    return {
        "metric_refinement": metric_records,
        "finest_level_thresholds": thresholds,
        "finest_level_observed": observed,
        "absolute_threshold_scope": (
            "ENFORCED_FULL_100_PERIOD" if profile.name == "full-100-period" else "RECORDED_NOT_ENFORCED_SMOKE"
        ),
        "accuracy_basis": accuracy_basis,
        "status": "OBSERVED_WITHIN_EXACT_FIXTURE_REFINEMENT_GATES",
    }


def _invariant_convergence(
    lanes: list[dict[str, Any]], profile: PrecisionProfile
) -> dict[str, Any]:
    if [lane["divisor"] for lane in lanes] != list(STEP_DIVISORS):
        raise PrecisionProbeError("invariant convergence lane order changed")
    paths = {
        "sampled_relative_energy_maximum_abs": (
            "metrics", "invariants", "relative_total_energy", "maximum_abs"
        ),
        "sampled_relative_energy_rms": (
            "metrics", "invariants", "relative_total_energy", "rms"
        ),
    }
    refinement: dict[str, Any] = {}
    for label, path in paths.items():
        values = [_metric_value(lane, path) for lane in lanes]
        orders = [math.log2(values[index] / values[index + 1]) for index in (0, 1)]
        if not values[0] > values[1] > values[2] or not all(order >= 1.8 for order in orders):
            raise PrecisionProbeError(f"{label} did not satisfy its sampled invariant gate")
        refinement[label] = {
            "values_p256_p512_p1024": values,
            "adjacent_orders": orders,
            "strictly_monotonic": True,
            "interpretation": "CIRCULAR_FIXTURE_SAMPLED_ENERGY_SUPERCONVERGENCE_NOT_METHOD_ORDER",
        }
    fine = lanes[-1]["metrics"]["invariants"]
    thresholds = {
        "sampled_relative_energy_maximum_abs": 1.0e-9,
        "relative_angular_momentum_vector_norm_maximum": 1.0e-12,
        "relative_angular_momentum_z_signed_maximum_abs": 1.0e-12,
        "center_of_mass_position_norm_maximum": 1.0e-12,
        "total_momentum_norm_maximum": 1.0e-12,
    }
    observed = {
        "sampled_relative_energy_maximum_abs": fine["relative_total_energy"]["maximum_abs"],
        "relative_angular_momentum_vector_norm_maximum": fine["relative_angular_momentum_vector_norm"]["maximum"],
        "relative_angular_momentum_z_signed_maximum_abs": fine["relative_angular_momentum_z_signed"]["maximum_abs"],
        "center_of_mass_position_norm_maximum": fine["center_of_mass_position_norm"]["maximum"],
        "total_momentum_norm_maximum": fine["total_momentum_norm"]["maximum"],
    }
    if profile.name == "full-100-period" and any(
        observed[key] > threshold for key, threshold in thresholds.items()
    ):
        raise PrecisionProbeError("the finest full-profile invariant exceeded an absolute gate")
    return {
        "metric_refinement": refinement,
        "finest_level_thresholds": thresholds,
        "finest_level_observed": observed,
        "absolute_threshold_scope": (
            "ENFORCED_FULL_100_PERIOD"
            if profile.name == "full-100-period"
            else "RECORDED_NOT_ENFORCED_SMOKE"
        ),
        "sampling_scope": "RETAINED_OUTPUT_CHECKPOINTS_ONLY_NOT_CONTINUOUS_TIME",
        "status": "OBSERVED_WITHIN_EXACT_FIXTURE_SAMPLED_INVARIANT_GATES",
    }


def _family_convergence(
    lanes: list[dict[str, Any]], profile: PrecisionProfile
) -> dict[str, Any]:
    return {
        "primary_accuracy_basis": "DECLARED_GRID_FIXED_HORIZON_END_TO_END_ACCURACY",
        "secondary_accuracy_basis": "ANALYTIC_STATE_AT_EACH_LANE_OBSERVED_CLOCK_CONDITIONAL_ON_EFFECTIVE_READBACK_TIME",
        "observed_clock_analytic": _convergence_summary(
            lanes,
            profile,
            accuracy_basis="observed_clock_analytic_accuracy",
        ),
        "declared_grid_delivery_aware": _convergence_summary(
            lanes,
            profile,
            accuracy_basis="declared_grid_analytic_accuracy",
        ),
        "sampled_invariants": _invariant_convergence(lanes, profile),
    }


def _typed_tree(value: Any, *, depth: int = 0, counter: list[int] | None = None) -> Any:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
        raise PrecisionProbeError("typed semantic tree exceeds its cap")
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        if value.bit_length() > MAX_JSON_INTEGER_BITS:
            raise PrecisionProbeError("integer exceeds its cap")
        return ["int", str(value)]
    if type(value) is float:
        if not math.isfinite(value):
            raise PrecisionProbeError("semantic float is nonfinite")
        return ["binary64", value.hex()]
    if type(value) is str:
        if len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES:
            raise PrecisionProbeError("semantic string exceeds its cap")
        return ["string", value]
    if type(value) is list:
        return ["list", [_typed_tree(item, depth=depth + 1, counter=counter) for item in value]]
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise PrecisionProbeError("semantic object key is not a string")
        return [
            "map",
            [
                [key, _typed_tree(value[key], depth=depth + 1, counter=counter)]
                for key in sorted(value)
            ],
        ]
    raise PrecisionProbeError("semantic tree contains an unsupported type")


def _exact_tree_equal(left: Any, right: Any) -> bool:
    return _typed_tree(left) == _typed_tree(right)


def _exact_array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _semantic_sha256(report: dict[str, Any]) -> str:
    _validate_json_tree(report)
    payload = copy.deepcopy(report)
    payload.pop("content_integrity", None)
    payload.pop("execution_diagnostics", None)
    encoded = json.dumps(
        _typed_tree(payload),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(SEMANTIC_DOMAIN.encode("ascii") + b"\0" + encoded).hexdigest()


def _attach_integrity(report: dict[str, Any]) -> dict[str, Any]:
    if "content_integrity" in report:
        raise PrecisionProbeError("report already has integrity metadata")
    result = copy.deepcopy(report)
    result["content_integrity"] = {
        "algorithm": "SHA256_DOMAIN_SEPARATED_FULLY_TYPED_BINARY64_JSON_V1",
        "typed_tree_schema": TYPED_TREE_SCHEMA,
        "domain": SEMANTIC_DOMAIN,
        "excluded_top_level_fields": ["content_integrity", "execution_diagnostics"],
        "semantic_content_sha256": _semantic_sha256(result),
        "authentication_or_authority_claimed": False,
    }
    return result


def _validate_json_tree(value: Any) -> None:
    strings = [0]
    nodes = [0]

    def visit(item: Any, depth: int) -> None:
        nodes[0] += 1
        if nodes[0] > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise PrecisionProbeError("JSON tree exceeds its cap")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if item.bit_length() > MAX_JSON_INTEGER_BITS:
                raise PrecisionProbeError("JSON integer exceeds its cap")
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise PrecisionProbeError("JSON contains nonfinite float")
            return
        if type(item) is str:
            strings[0] += len(item.encode("utf-8"))
            if strings[0] > MAX_JSON_STRING_BYTES:
                raise PrecisionProbeError("JSON strings exceed their cap")
            return
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for key, child in item.items():
                strings[0] += len(key.encode("utf-8"))
                if strings[0] > MAX_JSON_STRING_BYTES:
                    raise PrecisionProbeError("JSON strings exceed their cap")
                visit(child, depth + 1)
            return
        raise PrecisionProbeError("JSON tree contains an unsupported type")

    visit(value, 0)


def _lane_from_matrix_summary(summary: dict[str, Any], profile: PrecisionProfile) -> np.ndarray:
    if (
        type(summary["lane_id"]) is not str
        or type(summary["family"]) is not str
        or type(summary["role"]) is not str
        or type(summary["method_id"]) is not str
        or type(summary["divisor"]) is not int
        or type(summary["fixed_step_hex"]) is not str
    ):
        raise PrecisionProbeError("lane scalar field type changed")
    matrix = _decode_matrix_evidence(
        summary["raw_state_evidence"],
        expected_rows=profile.checkpoint_count,
        expected_columns=MATRIX_COLUMNS,
    )
    expected_raw = EXPECTED_RAW_STATE_SHA256[profile.name].get(summary["lane_id"])
    if expected_raw is None or summary["raw_state_evidence"]["sha256"] != expected_raw:
        raise PrecisionProbeError("lane raw-state regression KAT changed")
    divisor = summary["divisor"]
    expected_declared = profile.declared_epochs()
    if not _exact_array_equal(
        np.ascontiguousarray(matrix[:, 0]), np.ascontiguousarray(expected_declared)
    ):
        raise PrecisionProbeError("lane declared grid changed")
    expected_initial_position, expected_initial_velocity = analytic_trajectory(np.array((0.0,), dtype="<f8"))
    if (
        not _exact_array_equal(
            np.ascontiguousarray(matrix[0, 2:8]),
            np.ascontiguousarray(expected_initial_position[0].reshape(6)),
        )
        or not _exact_array_equal(
            np.ascontiguousarray(matrix[0, 8:14]),
            np.ascontiguousarray(expected_initial_velocity[0].reshape(6)),
        )
    ):
        raise PrecisionProbeError("lane initial state changed")
    expected_id = f"{summary['family']}_p{divisor}"
    expected_method = (
        FIXED_STEP_KDK_METHOD_ID if summary["family"] == "jx_kdk" else "rebound.integrator.leapfrog.5.1.1"
    )
    expected_role = "JX_FIXED_GRID" if summary["family"] == "jx_kdk" else "OPTIONAL_EXTERNAL_COMPARATOR"
    if (
        type(divisor) is not int
        or divisor not in STEP_DIVISORS
        or summary["lane_id"] != expected_id
        or summary["method_id"] != expected_method
        or summary["role"] != expected_role
        or summary["fixed_step_hex"] != float(PERIOD / divisor).hex()
    ):
        raise PrecisionProbeError("lane identity or method changed")
    if not _exact_tree_equal(summary["metrics"], _lane_metrics(matrix)):
        raise PrecisionProbeError("lane metrics do not match retained raw state")
    replay = summary["replay"]
    if not _exact_tree_equal(replay, {
        "scope": "SAME_RUNTIME_COMPLETE_FRESH_STATE_REEXECUTION",
        "raw_state_sha256": summary["raw_state_evidence"]["sha256"],
        "status": "BITWISE_IDENTICAL",
        "process_independence_claimed": False,
    }):
        raise PrecisionProbeError("lane replay evidence changed")
    accounting = summary["accounting"]
    expected_steps = profile.periods * divisor
    if summary["family"] == "jx_kdk":
        if set(accounting) != {
            "checkpoint_count", "completed_steps", "force_evaluations", "schedule_content_sha256", "result_content_sha256", "engine_internal_trajectory_reexecution_count", "engine_result_constructor_validation_performed", "effective_settings"
        } or (
            accounting["checkpoint_count"] != profile.checkpoint_count
            or accounting["completed_steps"] != expected_steps
            or accounting["force_evaluations"] != expected_steps + 1
            or type(accounting["engine_internal_trajectory_reexecution_count"]) is not int
            or accounting["engine_internal_trajectory_reexecution_count"] != 0
            or type(accounting["engine_result_constructor_validation_performed"]) is not bool
            or accounting["engine_result_constructor_validation_performed"] is not True
            or not _is_sha256(accounting["schedule_content_sha256"])
            or not _is_sha256(accounting["result_content_sha256"])
            or {
                "schedule_content_sha256": accounting["schedule_content_sha256"],
                "result_content_sha256": accounting["result_content_sha256"],
            }
            != EXPECTED_JX_NATIVE_DIGESTS[profile.name][divisor]
            or not _exact_tree_equal(
                accounting["effective_settings"],
                _expected_jx_effective_settings(profile, divisor),
            )
            or not _exact_tree_equal(summary["runtime"], _jx_runtime_contract())
        ):
            raise PrecisionProbeError("JX accounting or runtime changed")
        if not _exact_array_equal(
            np.ascontiguousarray(matrix[:, 0]), np.ascontiguousarray(matrix[:, 1])
        ):
            raise PrecisionProbeError("JX observed clock changed")
    else:
        if set(accounting) != {
            "checkpoint_count", "completed_integer_steps", "simulation_steps_call_count", "simulation_steps_argument", "force_evaluations", "one_step_dkd_evidence", "effective_settings"
        } or (
            accounting["checkpoint_count"] != profile.checkpoint_count
            or accounting["completed_integer_steps"] != expected_steps
            or accounting["simulation_steps_call_count"] != profile.checkpoint_count - 1
            or accounting["simulation_steps_argument"] != divisor // SAMPLES_PER_PERIOD
            or accounting["force_evaluations"] is not None
            or not _exact_tree_equal(accounting["one_step_dkd_evidence"], {
                "equation": "BENCHMARK_LOCAL_BINARY64_DRIFT_KICK_DRIFT",
                "maximum_component_ulp_distance": EXPECTED_REBOUND_ONE_STEP_ULP[divisor],
                "maximum_allowed_component_ulp_distance": 4,
                "zero_and_signed_zero_policy": "EXACT_BINARY64_BITS",
                "bitwise_same_runtime_replay_required_separately": True,
                "qualification_authorized": False,
            })
            or not _exact_tree_equal(
                accounting["effective_settings"],
                _expected_rebound_effective_settings(divisor),
            )
            or not _exact_tree_equal(summary["runtime"], _reported_rebound_runtime_contract())
        ):
            raise PrecisionProbeError("REBOUND accounting or runtime changed")
        if np.any(np.diff(matrix[:, 1]) <= 0.0):
            raise PrecisionProbeError("REBOUND observed clock is not monotonic")
    return matrix


def _reported_rebound_runtime_contract() -> dict[str, Any]:
    expected = EXPECTED_REBOUND_PROFILE
    return {
        "version": expected["version"],
        "githash": expected["githash"],
        "module": {
            "relative_path": "rebound/__init__.py",
            "size_bytes": expected["module_size_bytes"],
            "sha256": expected["module_sha256"],
        },
        "library": {
            "relative_path": "librebound.cpython-314-x86_64-linux-gnu.so",
            "size_bytes": expected["library_size_bytes"],
            "sha256": expected["library_sha256"],
        },
        "metadata": {
            "relative_path": "rebound-5.1.1.dist-info/METADATA",
            "size_bytes": expected["metadata_size_bytes"],
            "sha256": expected["metadata_sha256"],
        },
        "license": {
            "relative_path": "rebound-5.1.1.dist-info/licenses/LICENSE",
            "size_bytes": expected["license_size_bytes"],
            "sha256": expected["license_sha256"],
        },
        "python_source_file_count": expected["python_source_file_count"],
        "python_source_tree_sha256": expected["python_source_tree_sha256"],
        "license_expression": "GPL-3.0-only",
        "authority_authorized": False,
    }


def validate_report(report: Any) -> None:
    _validate_json_tree(report)
    if type(report) is not dict or set(report) != {
        "schema", "benchmark_id", "profile", "fixture", "sources", "analytic_oracle", "analytic_reference", "external_status", "lanes", "convergence", "claim_controls", "scientific_interpretation", "serialization", "execution_diagnostics", "content_integrity"
    }:
        raise PrecisionProbeError("report schema changed")
    if report["schema"] != SCHEMA or report["benchmark_id"] != BENCHMARK_ID:
        raise PrecisionProbeError("report identity changed")
    profile_record = report["profile"]
    if type(profile_record) is not dict or set(profile_record) != {
        "name", "periods", "samples_per_period", "checkpoint_count", "step_divisors", "cli_full_profile_cost_acknowledgement_required"
    }:
        raise PrecisionProbeError("profile schema changed")
    profile = PrecisionProfile(profile_record["name"])
    if not _exact_tree_equal(profile_record, {
        "name": profile.name,
        "periods": profile.periods,
        "samples_per_period": SAMPLES_PER_PERIOD,
        "checkpoint_count": profile.checkpoint_count,
        "step_divisors": list(STEP_DIVISORS),
        "cli_full_profile_cost_acknowledgement_required": True,
    }):
        raise PrecisionProbeError("profile values changed")
    if not _exact_tree_equal(report["fixture"], {
        "model": "ANALYTIC_EQUAL_MASS_CIRCULAR_BINARY",
        "G": 1.0,
        "body_ids": list(BODY_IDS),
        "masses_and_gravitational_parameters": list(MASS_VALUES),
        "initial_positions": [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "initial_velocities": [[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
        "period_hex": PERIOD.hex(),
        "analytic_total_energy": ENERGY_0,
        "analytic_total_angular_momentum": list(ANGULAR_MOMENTUM_0),
    }):
        raise PrecisionProbeError("fixture changed")
    sources = report["sources"]
    if type(sources) is not dict or set(sources) != {
        "probe_source", "analytic_support_source", "package_initializer", "engine_source_manifest"
    }:
        raise PrecisionProbeError("source schema changed")
    current_probe = _file_identity(Path(__file__).resolve(), relative_path="benchmarks/jx_post_v5_precision_probe.py")
    if (
        not _exact_tree_equal(sources["probe_source"], current_probe)
        or not _exact_tree_equal(
            sources["analytic_support_source"], _support_source_identity()
        )
        or not _exact_tree_equal(
            sources["package_initializer"], _package_init_identity()
        )
    ):
        raise PrecisionProbeError("probe or support source identity changed")
    engine = sources["engine_source_manifest"]
    if (
        type(engine) is not dict
        or engine.get("tree_sha256") != EXPECTED_ENGINE_SOURCE_TREE_SHA256
        or not _exact_tree_equal(engine, _engine_source_manifest())
    ):
        raise PrecisionProbeError("engine source identity changed")
    if not _exact_tree_equal(report["analytic_oracle"], _analytic_oracle_contract()):
        raise PrecisionProbeError("analytic oracle contract changed")
    analytic_matrix = _decode_matrix_evidence(
        report["analytic_reference"],
        expected_rows=profile.checkpoint_count,
        expected_columns=ANALYTIC_MATRIX_COLUMNS,
    )
    if (
        report["analytic_reference"]["sha256"]
        != EXPECTED_ANALYTIC_MATRIX_SHA256[profile.name]
    ):
        raise PrecisionProbeError("analytic reference regression KAT changed")
    expected_epochs = profile.declared_epochs()
    if not _exact_array_equal(analytic_matrix, _analytic_matrix(expected_epochs)):
        raise PrecisionProbeError("analytic reference bytes changed")
    status = report["external_status"]
    if type(status) is not dict or set(status) != {
        "requested_mode", "required_version", "external_lanes_executed", "reason", "external_code_vendored", "authority_authorized"
    }:
        raise PrecisionProbeError("external status schema changed")
    mode = status["requested_mode"]
    if (
        type(mode) is not str
        or mode not in REBOUND_MODES
        or type(status["required_version"]) is not str
        or status["required_version"] != REQUIRED_REBOUND_VERSION
        or type(status["reason"]) is not str
    ):
        raise PrecisionProbeError("external mode or version changed")
    executed = status["external_lanes_executed"]
    if type(executed) is not bool or status["external_code_vendored"] is not False or status["authority_authorized"] is not False:
        raise PrecisionProbeError("external claim changed")
    allowed_external_states = {
        ("disabled", False, "DISABLED_BY_CALLER"),
        ("auto", False, "EXACT_REBOUND_5_1_1_UNAVAILABLE"),
        ("auto", True, "EXACT_REBOUND_5_1_1_EXECUTED"),
        ("required", True, "EXACT_REBOUND_5_1_1_EXECUTED"),
    }
    if (mode, executed, status["reason"]) not in allowed_external_states:
        raise PrecisionProbeError("external execution state changed")
    lanes = report["lanes"]
    expected_lane_ids = [f"jx_kdk_p{value}" for value in STEP_DIVISORS]
    if executed:
        expected_lane_ids.extend(f"rebound_leapfrog_p{value}" for value in STEP_DIVISORS)
    if type(lanes) is not list or [item.get("lane_id") for item in lanes if type(item) is dict] != expected_lane_ids:
        raise PrecisionProbeError("lane roster changed")
    matrices: dict[str, np.ndarray] = {}
    for lane in lanes:
        if type(lane) is not dict or set(lane) != {
            "lane_id", "family", "role", "method_id", "divisor", "fixed_step_hex", "raw_state_evidence", "metrics", "accounting", "runtime", "replay"
        }:
            raise PrecisionProbeError("lane schema changed")
        matrices[lane["lane_id"]] = _lane_from_matrix_summary(lane, profile)
    families = {
        "jx_kdk": [lane for lane in lanes if lane["family"] == "jx_kdk"],
    }
    if executed:
        families["rebound_leapfrog"] = [lane for lane in lanes if lane["family"] == "rebound_leapfrog"]
    expected_convergence = {
        family: _family_convergence(items, profile) for family, items in families.items()
    }
    if not _exact_tree_equal(report["convergence"], expected_convergence):
        raise PrecisionProbeError("convergence summary changed")
    if not _exact_tree_equal(report["claim_controls"], CLAIM_CONTROLS):
        raise PrecisionProbeError("claim controls changed")
    expected_interpretation = {
        "allowed_claim": "EXACT_FIXTURE_DECLARED_GRID_PRIMARY_AND_OBSERVED_CLOCK_CONDITIONAL_MONOTONIC_REFINEMENT_WITH_FIXTURE_SPECIFIC_EMPIRICAL_SECOND_ORDER_STATE_PHASE_CONVERGENCE",
        "sampled_energy_conservation_implies_phase_or_state_correctness": False,
        "energy_slope_is_a_physical_secular_drift": False,
        "dyadic_ladder_rules_out_step_size_resonance": False,
        "scope": "THIS_ANALYTIC_BINARY_THE_THREE_DECLARED_STEP_LEVELS_AND_RETAINED_CHECKPOINTS_ONLY",
    }
    if not _exact_tree_equal(report["scientific_interpretation"], expected_interpretation):
        raise PrecisionProbeError("scientific interpretation changed")
    if not _exact_tree_equal(report["serialization"], {
        "report_format": "SORTED_FINITE_JSON",
        "raw_matrix_encoding": "RFC4648_BASE64_OF_C_ORDER_LITTLE_ENDIAN_BINARY64",
        "semantic_float_encoding": "BUILT_IN_FLOAT_HEX_IN_FULLY_TYPED_TREE",
        "maximum_report_bytes": MAX_REPORT_BYTES,
        "raw_state_regression_kat_scope": "EXACT_PROFILE_SOURCE_ENGINE_NUMPY_AND_OPTIONAL_REBOUND_BUILD",
        "timing_excluded_from_semantic_digest": True,
    }):
        raise PrecisionProbeError("serialization contract changed")
    diagnostics = report["execution_diagnostics"]
    if type(diagnostics) is not dict or set(diagnostics) != {
        "python_implementation", "python_version", "platform", "lane_timings", "timing_comparable", "process_independence_claimed"
    } or (
        diagnostics["timing_comparable"] is not False
        or diagnostics["process_independence_claimed"] is not False
        or type(diagnostics["lane_timings"]) is not list
        or len(diagnostics["lane_timings"]) != len(lanes)
    ):
        raise PrecisionProbeError("execution diagnostics changed")
    for field in ("python_implementation", "python_version", "platform"):
        value = diagnostics[field]
        if (
            type(value) is not str
            or not value
            or len(value.encode("utf-8")) > 1_024
            or any(ord(character) < 0x20 for character in value)
        ):
            raise PrecisionProbeError("execution diagnostic text is invalid")
    for item, lane in zip(diagnostics["lane_timings"], lanes, strict=True):
        if type(item) is not dict or set(item) != {
            "lane_id", "primary_wall_seconds", "replay_wall_seconds"
        } or item["lane_id"] != lane["lane_id"]:
            raise PrecisionProbeError("lane timing schema changed")
        for field in ("primary_wall_seconds", "replay_wall_seconds"):
            if type(item[field]) is not float or not math.isfinite(item[field]) or item[field] < 0.0:
                raise PrecisionProbeError("lane timing is invalid")
    integrity = report["content_integrity"]
    if type(integrity) is not dict or not _exact_tree_equal(integrity, {
        "algorithm": "SHA256_DOMAIN_SEPARATED_FULLY_TYPED_BINARY64_JSON_V1",
        "typed_tree_schema": TYPED_TREE_SCHEMA,
        "domain": SEMANTIC_DOMAIN,
        "excluded_top_level_fields": ["content_integrity", "execution_diagnostics"],
        "semantic_content_sha256": _semantic_sha256(report),
        "authentication_or_authority_claimed": False,
    }):
        raise PrecisionProbeError("semantic integrity changed")


def run_precision_probe(
    profile: PrecisionProfile = PrecisionProfile(),
    rebound_mode: str = "disabled",
) -> dict[str, Any]:
    if type(profile) is not PrecisionProfile:
        raise TypeError("profile must be an exact PrecisionProfile")
    profile.__post_init__()
    if type(rebound_mode) is not str or rebound_mode not in REBOUND_MODES:
        raise ValueError(f"rebound_mode must be one of {REBOUND_MODES!r}")
    probe_before = _file_identity(Path(__file__).resolve(), relative_path="benchmarks/jx_post_v5_precision_probe.py")
    support = _support_source_identity()
    package_initializer = _package_init_identity()
    engine = _engine_source_manifest()
    lanes: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for divisor in STEP_DIVISORS:
        summary, diagnostic = _lane_summary(
            _run_jx_lane(profile, divisor),
            _run_jx_lane(profile, divisor),
        )
        lanes.append(summary)
        timings.append(diagnostic)
    rebound: Any | None = None
    rebound_runtime: dict[str, Any] | None = None
    external_reason = "DISABLED_BY_CALLER"
    if rebound_mode != "disabled":
        try:
            rebound, rebound_runtime = _load_exact_rebound()
        except ReboundUnavailable:
            if rebound_mode == "required":
                raise
            external_reason = "EXACT_REBOUND_5_1_1_UNAVAILABLE"
        else:
            external_reason = "EXACT_REBOUND_5_1_1_EXECUTED"
    if rebound is not None and rebound_runtime is not None:
        for divisor in STEP_DIVISORS:
            summary, diagnostic = _lane_summary(
                _run_rebound_lane(profile, divisor, rebound, rebound_runtime),
                _run_rebound_lane(profile, divisor, rebound, rebound_runtime),
            )
            lanes.append(summary)
            timings.append(diagnostic)
    families = {
        "jx_kdk": [lane for lane in lanes if lane["family"] == "jx_kdk"],
    }
    if rebound is not None:
        families["rebound_leapfrog"] = [lane for lane in lanes if lane["family"] == "rebound_leapfrog"]
    declared = profile.declared_epochs()
    report = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "profile": {
            "name": profile.name,
            "periods": profile.periods,
            "samples_per_period": SAMPLES_PER_PERIOD,
            "checkpoint_count": profile.checkpoint_count,
            "step_divisors": list(STEP_DIVISORS),
            "cli_full_profile_cost_acknowledgement_required": True,
        },
        "fixture": {
            "model": "ANALYTIC_EQUAL_MASS_CIRCULAR_BINARY",
            "G": 1.0,
            "body_ids": list(BODY_IDS),
            "masses_and_gravitational_parameters": list(MASS_VALUES),
            "initial_positions": [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            "initial_velocities": [[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
            "period_hex": PERIOD.hex(),
            "analytic_total_energy": ENERGY_0,
            "analytic_total_angular_momentum": list(ANGULAR_MOMENTUM_0),
        },
        "sources": {
            "probe_source": probe_before,
            "analytic_support_source": support,
            "package_initializer": package_initializer,
            "engine_source_manifest": engine,
        },
        "analytic_oracle": _analytic_oracle_contract(),
        "analytic_reference": _matrix_evidence(_analytic_matrix(declared), ANALYTIC_MATRIX_COLUMNS),
        "external_status": {
            "requested_mode": rebound_mode,
            "required_version": REQUIRED_REBOUND_VERSION,
            "external_lanes_executed": rebound is not None,
            "reason": external_reason,
            "external_code_vendored": False,
            "authority_authorized": False,
        },
        "lanes": lanes,
        "convergence": {
            family: _family_convergence(items, profile) for family, items in families.items()
        },
        "claim_controls": dict(CLAIM_CONTROLS),
        "scientific_interpretation": {
            "allowed_claim": "EXACT_FIXTURE_DECLARED_GRID_PRIMARY_AND_OBSERVED_CLOCK_CONDITIONAL_MONOTONIC_REFINEMENT_WITH_FIXTURE_SPECIFIC_EMPIRICAL_SECOND_ORDER_STATE_PHASE_CONVERGENCE",
            "sampled_energy_conservation_implies_phase_or_state_correctness": False,
            "energy_slope_is_a_physical_secular_drift": False,
            "dyadic_ladder_rules_out_step_size_resonance": False,
            "scope": "THIS_ANALYTIC_BINARY_THE_THREE_DECLARED_STEP_LEVELS_AND_RETAINED_CHECKPOINTS_ONLY",
        },
        "serialization": {
            "report_format": "SORTED_FINITE_JSON",
            "raw_matrix_encoding": "RFC4648_BASE64_OF_C_ORDER_LITTLE_ENDIAN_BINARY64",
            "semantic_float_encoding": "BUILT_IN_FLOAT_HEX_IN_FULLY_TYPED_TREE",
            "maximum_report_bytes": MAX_REPORT_BYTES,
            "raw_state_regression_kat_scope": "EXACT_PROFILE_SOURCE_ENGINE_NUMPY_AND_OPTIONAL_REBOUND_BUILD",
            "timing_excluded_from_semantic_digest": True,
        },
        "execution_diagnostics": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "lane_timings": timings,
            "timing_comparable": False,
            "process_independence_claimed": False,
        },
    }
    result = _attach_integrity(report)
    if (
        _file_identity(Path(__file__).resolve(), relative_path="benchmarks/jx_post_v5_precision_probe.py")
        != probe_before
        or _support_source_identity() != support
        or _package_init_identity() != package_initializer
        or _engine_source_manifest() != engine
    ):
        raise PrecisionProbeError("source changed during precision execution")
    validate_report(result)
    encoded = json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii")
    if len(encoded) > MAX_REPORT_BYTES:
        raise PrecisionProbeError("precision report exceeds the byte cap")
    return result


def load_report_bytes(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes or not data or len(data) > MAX_REPORT_BYTES:
        raise PrecisionProbeError("report bytes are empty or exceed the cap")
    try:
        text = data.decode("ascii")
        value = json.loads(
            text,
            parse_int=_parse_json_integer,
            parse_float=_parse_json_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PrecisionProbeError(f"nonfinite JSON token {value!r}")
            ),
            object_pairs_hook=lambda pairs: _unique_object(pairs),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PrecisionProbeError("report is not strict ASCII JSON") from exc
    if type(value) is not dict:
        raise PrecisionProbeError("report root is not an object")
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii")
    if data not in (canonical, canonical + b"\n"):
        raise PrecisionProbeError("report bytes are not canonical sorted JSON")
    validate_report(value)
    return value


def _parse_json_integer(token: str) -> int:
    if type(token) is not str or len(token) > 1_235:
        raise PrecisionProbeError("JSON integer token exceeds its lexical cap")
    value = int(token, 10)
    if value.bit_length() > MAX_JSON_INTEGER_BITS:
        raise PrecisionProbeError("JSON integer exceeds its bit cap")
    return value


def _parse_json_float(token: str) -> float:
    if type(token) is not str or len(token) > 128:
        raise PrecisionProbeError("JSON float token exceeds its lexical cap")
    value = float(token)
    if not math.isfinite(value):
        raise PrecisionProbeError("JSON float is not finite binary64")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PrecisionProbeError("duplicate JSON object key")
        result[key] = value
    return result


def _write_new_file(path: Path, data: bytes) -> None:
    if (
        not isinstance(path, Path)
        or type(data) is not bytes
        or not data
        or len(data) > MAX_REPORT_BYTES + 1
    ):
        raise PrecisionProbeError("output path or bytes are invalid")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise PrecisionProbeError("output parent could not be resolved") from exc
    target = parent / path.name
    if (
        path.name in ("", ".", "..")
        or not parent.is_dir()
        or os.path.lexists(target)
    ):
        raise PrecisionProbeError("output parent must exist and target must not exist")
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.stage-", dir=parent
        )
    except OSError as exc:
        raise PrecisionProbeError("output staging file could not be created") from exc
    temporary = Path(temporary_name)
    directory_fd = -1
    published = False
    committed = False
    failure: BaseException | None = None
    try:
        directory_fd = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise PrecisionProbeError("output write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, target)
        published = True
        temporary.unlink()
        os.fsync(directory_fd)
        os.close(directory_fd)
        directory_fd = -1
        committed = True
    except (OSError, PrecisionProbeError) as exc:
        failure = exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                if failure is None:
                    failure = exc
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            if failure is None:
                failure = exc
        if published and not committed:
            try:
                target.unlink()
                if directory_fd >= 0:
                    os.fsync(directory_fd)
            except OSError as exc:
                failure = PrecisionProbeError(
                    f"output rollback failed after publication error: {exc}"
                )
        if directory_fd >= 0:
            try:
                os.close(directory_fd)
            except OSError as exc:
                if failure is None:
                    failure = exc
    if failure is not None:
        raise PrecisionProbeError("output could not be atomically published") from failure


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the additive JX energy/phase convergence probe.")
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="smoke")
    parser.add_argument("--rebound-mode", choices=REBOUND_MODES, default="disabled")
    parser.add_argument("--acknowledge-full-cost", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.profile == "full-100-period" and not arguments.acknowledge_full_cost:
        raise PrecisionProbeError("full profile requires --acknowledge-full-cost")
    report = run_precision_probe(PrecisionProfile(arguments.profile), arguments.rebound_mode)
    data = json.dumps(
        report,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    if arguments.output is None:
        sys.stdout.buffer.write(data)
    else:
        _write_new_file(arguments.output, data)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrecisionProbeError as exc:
        print(f"precision probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
