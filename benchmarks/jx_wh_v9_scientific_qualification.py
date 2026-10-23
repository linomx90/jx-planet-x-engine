#!/usr/bin/env python3
"""Closed, create-only executor for the registered V9 qualification protocol.

Passing this program is deliberately *not* scientific approval.  A complete
formal run can emit only ``ELIGIBLE_FOR_INDEPENDENT_REVIEW``; every incomplete
or failing run is ``INVALID_FOR_DECLARED_QUALIFICATION``.  The registered
protocol additionally requires an independently administered second host and
a named external reviewer before any approval claim.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import gzip
import io
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import types
from typing import Any, Iterable, Iterator
import zipfile

import numpy as np
from scipy import __version__ as SCIPY_VERSION
from scipy.integrate import solve_ivp

from benchmarks import rebound_whfast_comparison as fixture
from jxplanetx import __version__ as JX_VERSION
from jxplanetx import fast_wisdom_holman_v9 as v9
from jxplanetx.engine import (
    BackendSpec,
    FixedStepWisdomHolmanSpec,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "runs/jx_wh_v9_scientific_qualification_v1/protocol_v1.json"
PROTOCOL_SHA256 = "d9c94b04266519a40e102f7feb443439612ce1bfd89871fe09be652d51187512"
PROTOCOL_REGISTRATION_COMMIT = "65986fabbce5c9528fe7193cd01ed530afbcfcfa"
SCHEMA = "jxplanetx.wisdom-holman-v9-scientific-qualification-result.v1"
REPORT_DOMAIN = "jxplanetx.wisdom-holman-v9-scientific-qualification-result.v1"
ARRAY_DOMAIN = "jxplanetx.wisdom-holman-v9-qualification-array.v1"
SOURCE_CLOSURE_DOMAIN = "jxplanetx.wisdom-holman-v9-qualification-source-closure.v1"
ENGINEERING_SCHEMA = "jxplanetx.wisdom-holman-v9-engineering-receipt.v2"
DOP853_RECEIPT_DOMAIN = "jxplanetx.wh-v9.dop853-receipt.v1"
IAS15_RECEIPT_DOMAIN = "jxplanetx.wh-v9.ias15-receipt.v1"
MAX_REPORT_BYTES = 96 * 1024 * 1024
MAX_JSON_DEPTH = 80
MAX_JSON_NODES = 1_000_000
MAX_ENGINEERING_LOG_BYTES = 16 * 1024 * 1024
MAX_ENGINEERING_JSON_BYTES = 48 * 1024 * 1024
MAX_WHEEL_BYTES = 128 * 1024 * 1024
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_ARCHIVE_EXPANDED_BYTES = 256 * 1024 * 1024
RELEASE_SOURCE_DATE_EPOCH = 1_700_000_000
DECIMAL_PRECISION = 100
DECIMAL_WORK_PRECISION = 116
EXPECTED_COMPILER_VERSION = "15.2.0"
DECIMAL_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406"
    "286208998628034825342117067982148086513282306647093844609550582231725359"
)
FORMAL = "formal"
SMOKE = "smoke"
PROFILE_NAMES = (SMOKE, FORMAL)
INTERNAL_PASS = "ELIGIBLE_FOR_INDEPENDENT_REVIEW"
INTERNAL_FAIL = "INVALID_FOR_DECLARED_QUALIFICATION"
V9_NATIVE_SOURCE_SHA256 = {
    "_wisdom_holman_cpu_v9_module.c": "90468c9f527eefb6347b31e8e371310d1e054e2703bb6043baaae71d4078d654",
    "_wisdom_holman_cpu_v8_module.c": "fcf77481d241902f2c05eabaaac9bfa89f208e645d9ae760baf67dcd03e9d288",
    "_wisdom_holman_loop_v9.c": "680b4f887fa82561bb503a7982367f68c9c6648cd67ed8b926f1ae87e0d41bd2",
    "_wisdom_holman_loop_v8.c": "6fe8b4fcda8947bc21aacce56d9e3383a1d5d579176c742a994628967927b004",
    "_wisdom_holman_loop_v7.c": "337c92c1ea4aba54f64e8e336889a02dccb23d2bd6c275dedfac39e73fd9c0bb",
    "_wisdom_holman_loop_v6.c": "c1da285ec602185f093c7f702097650f009827285332e2aeb1c5f06aa848590d",
    "_wisdom_holman_loop_v4.c": "10981b416e4bb90a89babfc05cdd9c97bf1e5b49a0956490300bc5cbc591a16c",
    "_wisdom_holman_loop_v3.c": "7143fe32304b04901d107b48bf8fe5ae9043118b81c25c635e5d4b412c93b629",
    "_wisdom_holman_loop_v2.c": "38b8898bc6c1358522a0f653fc55849719e955889f556fd9f5850ad199346750",
}
REQUIRED_SOURCE_PATHS = (
    "pyproject.toml",
    "MANIFEST.in",
    "benchmarks/jx_wh_v9_scientific_qualification.py",
    "benchmarks/rebound_leapfrog_comparison.py",
    "benchmarks/rebound_whfast_comparison.py",
    "runs/jx_wh_v9_scientific_qualification_v1/protocol_v1.json",
    "src/jxplanetx/engine/contracts.py",
    "src/jxplanetx/engine/wisdom_holman.py",
    "src/jxplanetx/engine/wisdom_holman_contracts.py",
    "src/jxplanetx/fast_wisdom_holman.py",
    "src/jxplanetx/fast_wisdom_holman_v9.py",
    *(f"src/jxplanetx/{name}" for name in V9_NATIVE_SOURCE_SHA256),
    "tools/build_release.py",
    "tests/test_wisdom_holman_v9.py",
    "tests/test_fast_wisdom_holman_v9.py",
    "tests/test_jx_wh_v9_scientific_qualification.py",
)
ENGINEERING_TEST_MODULES = (
    "tests.test_wisdom_holman_v9",
    "tests.test_fast_wisdom_holman_v9",
    "tests.test_jx_wh_v9_scientific_qualification",
)
ENGINEERING_TOOL_VERSIONS = {
    "python": "3.14.4",
    "setuptools": "78.1.1",
    "wheel": "0.48.0",
    "packaging": "26.3",
}
FORMAL_BUILD_PATH = "/usr/bin:/bin"


class QualificationError(RuntimeError):
    """The registered qualification contract could not be satisfied."""


@dataclass(frozen=True)
class EngineeringEvidencePaths:
    normal_log: Path
    optimized_log: Path
    wheel_build_a: Path
    wheel_build_b: Path
    sdist_build_a: Path
    sdist_build_b: Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return data + (b"\n" if newline else b"")


def _domain_sha256(domain: str, value: Any) -> str:
    return _sha256(domain.encode("ascii") + b"\0" + _canonical_bytes(value))


def _file_identity(path: Path, relative: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"custody source is unavailable: {relative}")
    data = path.read_bytes()
    if not data:
        raise QualificationError(f"custody source is empty: {relative}")
    return {"relative_path": relative, "size_bytes": len(data), "sha256": _sha256(data)}


def _protocol() -> dict[str, Any]:
    raw = PROTOCOL_PATH.read_bytes()
    if _sha256(raw) != PROTOCOL_SHA256:
        raise QualificationError("registered protocol bytes changed")
    value = json.loads(raw)
    if (
        type(value) is not dict
        or value.get("protocol_revision") != 3
        or value.get("qualification_id") != "JX_WH_V9_SCIENTIFIC_QUALIFICATION_V1"
    ):
        raise QualificationError("registered protocol identity changed")
    return value


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise QualificationError(f"git custody command failed: {' '.join(arguments)}")
    return completed.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments), cwd=ROOT, capture_output=True, check=False
    )
    if completed.returncode:
        raise QualificationError(f"git custody command failed: {' '.join(arguments)}")
    return completed.stdout


def _source_custody(*, require_clean: bool = False) -> dict[str, Any]:
    protocol_commit = _git("log", "-1", "--format=%H", "--", str(PROTOCOL_PATH.relative_to(ROOT)))
    if protocol_commit != PROTOCOL_REGISTRATION_COMMIT:
        raise QualificationError("protocol registration commit changed")
    execution_head = _git("rev-parse", "HEAD")
    clean = _git("status", "--short") == ""
    records: list[dict[str, Any]] = []
    for relative in REQUIRED_SOURCE_PATHS:
        file_record = _file_identity(ROOT / relative, relative)
        completed = subprocess.run(
            ("git", "rev-parse", f"{execution_head}:{relative}"),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            blob_id: str | None = completed.stdout.strip()
            blob_bytes = _git_bytes("cat-file", "blob", blob_id)
            tracked = True
            if blob_bytes != (ROOT / relative).read_bytes():
                tracked = False
        else:
            blob_id = None
            tracked = False
        records.append(
            file_record
            | {
                "git_blob_sha1": blob_id,
                "tracked_bytes_equal_head": tracked,
            }
        )
    tree_verified = clean and all(record["tracked_bytes_equal_head"] for record in records)
    if require_clean and not tree_verified:
        raise QualificationError(
            "formal execution requires a clean HEAD with every qualification source tracked"
        )
    closure_payload = {
        "execution_head": execution_head,
        "files": records,
    }
    return {
        "protocol_registration_commit": protocol_commit,
        "execution_head": execution_head,
        "worktree_clean": clean,
        "git_tree_verified": tree_verified,
        "custody_mode": (
            "CLEAN_GIT_HEAD_TREE_VERIFIED"
            if tree_verified
            else "UNCOMMITTED_SMOKE_ONLY"
        ),
        "files": records,
        "closure_algorithm": "SHA256_DOMAIN_SEPARATED_CANONICAL_ASCII_JSON_V1",
        "closure_domain": SOURCE_CLOSURE_DOMAIN,
        "closure_sha256": _domain_sha256(SOURCE_CLOSURE_DOMAIN, closure_payload),
    }


def _validate_file_identity_record(value: Any, relative_path: str, *, expected_sha256: str | None = None) -> None:
    if (
        type(value) is not dict
        or set(value) != {"relative_path", "size_bytes", "sha256"}
        or value["relative_path"] != relative_path
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] <= 0
        or not _is_sha256(value["sha256"])
        or (expected_sha256 is not None and value["sha256"] != expected_sha256)
    ):
        raise QualificationError(f"intrinsic file identity changed: {relative_path}")


def _validate_intrinsic_sources(value: Any, protocol: dict[str, Any], profile: str) -> None:
    if type(value) is not dict or set(value) != {
        "protocol_registration_commit", "execution_head", "worktree_clean", "git_tree_verified", "custody_mode", "files", "closure_algorithm", "closure_domain", "closure_sha256"
    }:
        raise QualificationError("source-custody schema changed")
    if (
        value["protocol_registration_commit"] != PROTOCOL_REGISTRATION_COMMIT
        or not _is_git_commit(value["execution_head"])
        or type(value["worktree_clean"]) is not bool
        or type(value["git_tree_verified"]) is not bool
        or value["closure_algorithm"] != "SHA256_DOMAIN_SEPARATED_CANONICAL_ASCII_JSON_V1"
        or value["closure_domain"] != SOURCE_CLOSURE_DOMAIN
    ):
        raise QualificationError("source-custody commit record changed")
    files = value["files"]
    if type(files) is not list or len(files) != len(REQUIRED_SOURCE_PATHS):
        raise QualificationError("source-closure roster changed")
    for expected_path, record in zip(REQUIRED_SOURCE_PATHS, files, strict=True):
        if type(record) is not dict or set(record) != {
            "relative_path", "size_bytes", "sha256", "git_blob_sha1", "tracked_bytes_equal_head"
        }:
            raise QualificationError("source-closure record schema changed")
        _validate_file_identity_record(
            {key: record[key] for key in ("relative_path", "size_bytes", "sha256")},
            expected_path,
            expected_sha256=(
                PROTOCOL_SHA256
                if expected_path.endswith("protocol_v1.json")
                else protocol["candidate"]["wrapper_sha256"]
                if expected_path == "src/jxplanetx/fast_wisdom_holman_v9.py"
                else V9_NATIVE_SOURCE_SHA256.get(Path(expected_path).name)
            ),
        )
        if (
            type(record["tracked_bytes_equal_head"]) is not bool
            or (
                record["git_blob_sha1"] is not None
                and not _is_git_commit(record["git_blob_sha1"])
            )
            or (
                record["tracked_bytes_equal_head"]
                and record["git_blob_sha1"] is None
            )
        ):
            raise QualificationError("source-closure git binding changed")
    closure_payload = {
        "execution_head": value["execution_head"],
        "files": files,
    }
    if value["closure_sha256"] != _domain_sha256(SOURCE_CLOSURE_DOMAIN, closure_payload):
        raise QualificationError("source-closure digest changed")
    expected_tree_verified = value["worktree_clean"] and all(
        record["tracked_bytes_equal_head"] for record in files
    )
    if (
        value["git_tree_verified"] is not expected_tree_verified
        or value["custody_mode"]
        != (
            "CLEAN_GIT_HEAD_TREE_VERIFIED"
            if expected_tree_verified
            else "UNCOMMITTED_SMOKE_ONLY"
        )
        or (profile == FORMAL and not expected_tree_verified)
    ):
        raise QualificationError("source-closure clean-tree assertion changed")


def _source_record(value: dict[str, Any], relative_path: str) -> dict[str, Any]:
    for record in value["files"]:
        if record["relative_path"] == relative_path:
            return record
    raise QualificationError(f"source record is missing: {relative_path}")


def _validate_intrinsic_runtime(value: Any, protocol: dict[str, Any]) -> None:
    if type(value) is not dict or set(value) != {
        "python", "platform", "numpy", "scipy", "jxplanetx_version", "v9"
    }:
        raise QualificationError("runtime-custody schema changed")
    python = value["python"]
    if (
        type(python) is not dict
        or set(python) != {"implementation", "version", "optimization_level", "executable_sha256"}
        or python["implementation"] != "CPython"
        or type(python["version"]) is not str
        or python["optimization_level"] not in (0, 1, 2)
        or not _is_sha256(python["executable_sha256"])
    ):
        raise QualificationError("Python runtime custody changed")
    if value["platform"] != {"system": "Linux", "machine": "x86_64", "byteorder": "little"}:
        raise QualificationError("platform runtime custody changed")
    for package in ("numpy", "scipy"):
        record = value[package]
        if type(record) is not dict or type(record.get("version")) is not str:
            raise QualificationError(f"{package} runtime custody changed")
        _validate_file_identity_record(record.get("module"), f"{package}/__init__.py")
    scipy = value["scipy"]
    if set(scipy) != {
        "version", "distribution_version", "module", "solve_ivp_source", "dop853_source"
    } or scipy["distribution_version"] != scipy["version"]:
        raise QualificationError("SciPy distribution custody changed")
    _validate_file_identity_record(
        scipy["solve_ivp_source"], "scipy/integrate/_ivp/ivp.py"
    )
    _validate_file_identity_record(
        scipy["dop853_source"], "scipy/integrate/_ivp/rk.py"
    )
    if set(value["numpy"]) != {"version", "module"}:
        raise QualificationError("NumPy runtime custody changed")
    if value["jxplanetx_version"] != protocol["candidate"]["package_version"]:
        raise QualificationError("JX package version changed")
    identity = value["v9"]
    required_identity = {
        "schema", "api_id", "result_type_id", "kernel_id", "g_kernel_id", "exact_fallback_kernel_id", "scientific_claim_state",
        "support_status", "public_default", "registry_authorized", "qualification_authorized", "production_authorized", "superiority_claimed",
        "maximum_bodies", "metric_count", "counter_count", "compiler_family", "compiler_version", "compiler_flags", "runtime_fp_environment",
        "platform_policy", "source_closure_sha256", "compiled_sources", "extension_binary", "content_sha256",
    }
    if type(identity) is not dict or set(identity) != required_identity:
        raise QualificationError("V9 runtime identity schema changed")
    if (
        identity["schema"] != v9.SCHEMA
        or identity["api_id"] != protocol["candidate"]["api_id"]
        or identity["result_type_id"] != v9.RESULT_TYPE_ID
        or identity["kernel_id"] != v9.KERNEL_ID
        or identity["g_kernel_id"] != v9.G_KERNEL_ID
        or identity["exact_fallback_kernel_id"] != v9.EXACT_FALLBACK_KERNEL_ID
        or identity["scientific_claim_state"] != "SCREENING_ONLY"
        or identity["support_status"] != "SUPPORTED_OPT_IN_LINUX_X86_64_GCC"
        or identity["public_default"] is not False
        or any(identity[name] is not False for name in ("registry_authorized", "qualification_authorized", "production_authorized", "superiority_claimed"))
        or identity["maximum_bodies"] != 32
        or identity["metric_count"] != len(v9.METRIC_NAMES)
        or identity["counter_count"] != len(v9.COUNTER_NAMES)
        or identity["compiler_family"] != "GCC"
        or identity["compiler_version"] != EXPECTED_COMPILER_VERSION
        or identity["compiler_flags"]
        != "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm"
        or identity["platform_policy"]
        != {
            "system": "Linux",
            "machine": "x86_64",
            "byteorder": "little",
            "floating_point": "IEEE_754_BINARY64",
            "rounding": "FE_TONEAREST",
            "mxcsr_all_exceptions_masked": True,
            "mxcsr_rounding_control": 0,
            "mxcsr_ftz": False,
            "mxcsr_daz": False,
        }
        or not _is_sha256(identity["source_closure_sha256"])
        or not _is_sha256(identity["content_sha256"])
    ):
        raise QualificationError("V9 runtime claim boundary changed")
    fp = identity["runtime_fp_environment"]
    if (
        type(fp) is not list
        or any(type(item) is not list or len(item) != 2 for item in fp)
        or tuple(item[0] for item in fp) != v9.FP_ENVIRONMENT_KEYS
    ):
        raise QualificationError("V9 floating-point environment schema changed")
    fp_environment = dict(fp)
    required_true = {
        "compiled_linux", "compiled_x86_64", "compiled_little_endian",
        "runtime_linux", "runtime_x86_64", "runtime_little_endian",
        "ieee_binary64_layout", "round_to_nearest", "mxcsr_available",
        "mxcsr_all_exceptions_masked", "mxcsr_round_to_nearest", "supported",
    }
    if (
        fp_environment["compiler_family"] != "GCC"
        or any(fp_environment[name] is not True for name in required_true)
        or fp_environment["rounding_mode"] != 0
        or fp_environment["fe_tonearest"] != 0
        or type(fp_environment["mxcsr"]) is not int
        or not 0 <= fp_environment["mxcsr"] <= 0xFFFFFFFF
        or fp_environment["mxcsr"] & 0x3F
        or fp_environment["mxcsr_rounding_control"] != 0
        or (fp_environment["mxcsr"] & 0x1F80) != 0x1F80
        or fp_environment["mxcsr_ftz"] is not False
        or fp_environment["mxcsr_daz"] is not False
    ):
        raise QualificationError("V9 floating-point environment changed")
    compiled = identity["compiled_sources"]
    if (
        type(compiled) is not list
        or len(compiled) != len(V9_NATIVE_SOURCE_SHA256)
    ):
        raise QualificationError("V9 compiled-source custody changed")
    for (name, expected), record in zip(
        V9_NATIVE_SOURCE_SHA256.items(), compiled, strict=True
    ):
        if type(record) is not dict or set(record) != {"name", "size_bytes", "sha256"} or record["sha256"] != expected or type(record["size_bytes"]) is not int or record["size_bytes"] <= 0:
            raise QualificationError(f"V9 compiled-source identity changed: {name}")
        if record["name"] != name:
            raise QualificationError("V9 compiled-source roster changed")
    if (
        V9_NATIVE_SOURCE_SHA256["_wisdom_holman_cpu_v9_module.c"]
        != protocol["candidate"]["native_wrapper_sha256"]
        or V9_NATIVE_SOURCE_SHA256["_wisdom_holman_loop_v9.c"]
        != protocol["candidate"]["native_loop_sha256"]
        or identity["source_closure_sha256"]
        != _domain_sha256(v9.SOURCE_CLOSURE_CHECKSUM_DOMAIN, compiled)
    ):
        raise QualificationError("V9 compiled-source closure changed")
    extension = identity["extension_binary"]
    if type(extension) is not dict or set(extension) != {"name", "size_bytes", "sha256"} or type(extension["name"]) is not str or "/" in extension["name"] or "\\" in extension["name"] or not extension["name"].startswith("_wisdom_holman_cpu_v9.") or not extension["name"].endswith(".so") or type(extension["size_bytes"]) is not int or extension["size_bytes"] <= 0 or not _is_sha256(extension["sha256"]):
        raise QualificationError("V9 extension-binary identity changed")
    runtime_core = {
        key: identity[key] for key in identity if key != "content_sha256"
    }
    if identity["content_sha256"] != _domain_sha256(
        v9.RUNTIME_IDENTITY_CHECKSUM_DOMAIN, runtime_core
    ):
        raise QualificationError("V9 runtime-identity digest changed")


def _validate_source_runtime_closure(
    sources: dict[str, Any], runtime: dict[str, Any]
) -> None:
    """Bind retained compiled-source identities to the recorded Git closure."""

    for compiled in runtime["v9"]["compiled_sources"]:
        source = _source_record(
            sources, f"src/jxplanetx/{compiled['name']}"
        )
        if (
            source["size_bytes"] != compiled["size_bytes"]
            or source["sha256"] != compiled["sha256"]
        ):
            raise QualificationError(
                f"V9 runtime/source closure differs: {compiled['name']}"
            )


def _require_unchanged_live_custody(
    sources: dict[str, Any], runtime: dict[str, Any], phase: str
) -> None:
    """Re-sample formal custody after executable work, never self-attest it."""

    if _source_custody(require_clean=True) != sources:
        raise QualificationError(f"formal source custody changed {phase}")
    if _runtime_custody() != runtime:
        raise QualificationError(f"formal runtime custody changed {phase}")


def _validate_loaded_source_module(
    module: Any, relative: str, sources: dict[str, Any]
) -> None:
    expected_path = (ROOT / relative).resolve(strict=True)
    loaded_name = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    spec_origin = None if spec is None else getattr(spec, "origin", None)
    if type(loaded_name) is not str or type(spec_origin) is not str:
        raise QualificationError(f"loaded source module is unbound: {relative}")
    loaded_path = Path(loaded_name).resolve(strict=True)
    origin_path = Path(spec_origin).resolve(strict=True)
    record = _source_record(sources, relative)
    raw, identity = _read_bounded_regular_file(
        loaded_path, max(record["size_bytes"], 1), f"loaded module {relative}"
    )
    expected_stat = expected_path.stat()
    if (
        loaded_path != expected_path
        or origin_path != expected_path
        or identity != (expected_stat.st_dev, expected_stat.st_ino)
        or len(raw) != record["size_bytes"]
        or _sha256(raw) != record["sha256"]
    ):
        raise QualificationError(f"loaded source module differs: {relative}")


def _validate_formal_loaded_source_modules(sources: dict[str, Any]) -> None:
    bindings = (
        (fixture, "benchmarks/rebound_whfast_comparison.py"),
        (
            fixture._PROVENANCE_SUPPORT,
            "benchmarks/rebound_leapfrog_comparison.py",
        ),
        (v9, "src/jxplanetx/fast_wisdom_holman_v9.py"),
        (v9._v3, "src/jxplanetx/fast_wisdom_holman.py"),
        (
            sys.modules[BackendSpec.__module__],
            "src/jxplanetx/engine/contracts.py",
        ),
        (
            sys.modules[FixedStepWisdomHolmanSpec.__module__],
            "src/jxplanetx/engine/wisdom_holman_contracts.py",
        ),
        (
            sys.modules[fixture.integrate_wisdom_holman_trajectory.__module__],
            "src/jxplanetx/engine/wisdom_holman.py",
        ),
    )
    for module, relative in bindings:
        _validate_loaded_source_module(module, relative, sources)


def validate_live_reproduction(
    report: Any,
    *,
    engineering_paths: EngineeringEvidencePaths | None = None,
) -> None:
    """Re-execute the retained profile and compare its complete closed report."""

    validate_report(report)
    if report["profile"] == FORMAL and engineering_paths is None:
        raise QualificationError(
            "formal live reproduction requires all engineering evidence paths"
        )
    if report["profile"] == SMOKE and engineering_paths is not None:
        raise QualificationError(
            "smoke live reproduction does not accept engineering evidence"
        )
    if report["sources"] != _source_custody(require_clean=report["profile"] == FORMAL):
        raise QualificationError("live source custody differs from retained receipt")
    if report["runtime"] != _runtime_custody():
        raise QualificationError("live runtime custody differs from retained receipt")
    ias_presence = {
        lane["ias15"] is not None
        for horizon in ("short", "long")
        for lane in report["weak_three_body"][horizon]["lanes"]
    }
    if len(ias_presence) != 1:
        raise QualificationError("retained IAS15 lane presence is inconsistent")
    reproduced = run_qualification(
        report["profile"],
        engineering_paths=engineering_paths,
        include_ias15=ias_presence.pop(),
    )
    if _canonical_bytes(reproduced) != _canonical_bytes(report):
        raise QualificationError(
            "live scientific reproduction differs from retained report"
        )


def _runtime_custody() -> dict[str, Any]:
    numpy_path = Path(np.__file__).resolve()
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "optimization_level": int(sys.flags.optimize),
            "executable_sha256": _sha256(Path(sys.executable).resolve().read_bytes()),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "numpy": {
            "version": np.__version__,
            "module": _file_identity(numpy_path, "numpy/__init__.py"),
        },
        "scipy": _scipy_runtime_custody(),
        "jxplanetx_version": JX_VERSION,
        "v9": _json_tree(v9.fast_wisdom_holman_v9_runtime_identity()),
    }


def _scipy_runtime_custody() -> dict[str, Any]:
    scipy_dist = importlib.metadata.distribution("scipy")
    scipy_module = __import__("scipy")
    from scipy.integrate._ivp import ivp as scipy_ivp
    from scipy.integrate._ivp import rk as scipy_rk

    return {
        "version": SCIPY_VERSION,
        "distribution_version": scipy_dist.version,
        "module": _file_identity(
            Path(scipy_module.__file__).resolve(), "scipy/__init__.py"
        ),
        "solve_ivp_source": _file_identity(
            Path(scipy_ivp.__file__).resolve(), "scipy/integrate/_ivp/ivp.py"
        ),
        "dop853_source": _file_identity(
            Path(scipy_rk.__file__).resolve(), "scipy/integrate/_ivp/rk.py"
        ),
    }


def _json_tree(value: Any) -> Any:
    if type(value) is dict:
        return {key: _json_tree(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_json_tree(item) for item in value]
    return value


def _array_evidence(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    if not np.all(np.isfinite(array)):
        raise QualificationError("nonfinite array evidence")
    raw = array.tobytes(order="C")
    metadata = {"dtype": "<f8", "shape": list(array.shape), "byte_order": "LITTLE_ENDIAN_C_ORDER"}
    return metadata | {
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": _sha256(ARRAY_DOMAIN.encode("ascii") + b"\0" + _canonical_bytes(metadata) + b"\0" + raw),
    }


def _decode_array(record: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if type(record) is not dict or set(record) != {
        "dtype", "shape", "byte_order", "data_base64", "sha256"
    }:
        raise QualificationError("array-evidence schema changed")
    if record["dtype"] != "<f8" or record["byte_order"] != "LITTLE_ENDIAN_C_ORDER":
        raise QualificationError("array-evidence representation changed")
    if type(record["shape"]) is not list or any(type(v) is not int or v < 0 for v in record["shape"]):
        raise QualificationError("array-evidence shape is invalid")
    expected_shape = tuple(record["shape"])
    if shape is not None and expected_shape != shape:
        raise QualificationError("array-evidence shape changed")
    try:
        raw = base64.b64decode(record["data_base64"], validate=True)
    except Exception as exc:
        raise QualificationError("array-evidence base64 is invalid") from exc
    metadata = {"dtype": "<f8", "shape": list(expected_shape), "byte_order": "LITTLE_ENDIAN_C_ORDER"}
    expected = _sha256(ARRAY_DOMAIN.encode("ascii") + b"\0" + _canonical_bytes(metadata) + b"\0" + raw)
    if record["sha256"] != expected or len(raw) != math.prod(expected_shape) * 8:
        raise QualificationError("array-evidence bytes changed")
    array = np.frombuffer(raw, dtype="<f8").reshape(expected_shape).copy()
    if not np.all(np.isfinite(array)):
        raise QualificationError("array evidence contains nonfinite values")
    array.setflags(write=False)
    return array


def _array_bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _decimal_sin_cos(angle: Decimal) -> tuple[Decimal, Decimal]:
    with localcontext(Context(prec=DECIMAL_WORK_PRECISION, rounding=ROUND_HALF_EVEN)):
        two_pi = DECIMAL_PI * 2
        x = angle % two_pi
        if x > DECIMAL_PI:
            x -= two_pi
        x2 = x * x
        sine = x
        cosine = Decimal(1)
        sine_term = x
        cosine_term = Decimal(1)
        threshold = Decimal(1).scaleb(-(DECIMAL_PRECISION + 8))
        for index in range(1, 512):
            sine_term *= -x2 / Decimal((2 * index) * (2 * index + 1))
            cosine_term *= -x2 / Decimal((2 * index - 1) * (2 * index))
            sine += sine_term
            cosine += cosine_term
            if abs(sine_term) < threshold and abs(cosine_term) < threshold:
                return +sine, +cosine
    raise QualificationError("Decimal trigonometric series did not converge")


def _decimal_atan(value: Decimal) -> Decimal:
    with localcontext(Context(prec=DECIMAL_WORK_PRECISION, rounding=ROUND_HALF_EVEN)):
        if value == 0:
            return Decimal(0)
        sign = Decimal(-1) if value < 0 else Decimal(1)
        x = abs(value)
        reciprocal = x > 1
        if reciprocal:
            x = 1 / x
        reductions = 0
        while x > Decimal("0.1"):
            x = x / (Decimal(1) + (Decimal(1) + x * x).sqrt())
            reductions += 1
        x2 = x * x
        total = x
        term_power = x
        threshold = Decimal(1).scaleb(-(DECIMAL_PRECISION + 8))
        for index in range(1, 2048):
            term_power *= -x2
            term = term_power / Decimal(2 * index + 1)
            total += term
            if abs(term) < threshold:
                total *= Decimal(2) ** reductions
                if reciprocal:
                    total = DECIMAL_PI / 2 - total
                return +(sign * total)
    raise QualificationError("Decimal atan series did not converge")


def _decimal_atan2(y: Decimal, x: Decimal) -> Decimal:
    if x > 0:
        return _decimal_atan(y / x)
    if x < 0 and y >= 0:
        return _decimal_atan(y / x) + DECIMAL_PI
    if x < 0:
        return _decimal_atan(y / x) - DECIMAL_PI
    if y > 0:
        return DECIMAL_PI / 2
    if y < 0:
        return -DECIMAL_PI / 2
    raise QualificationError("atan2 is undefined at the origin")


def _solve_decimal_kepler(mean_anomaly: Decimal, eccentricity: Decimal) -> Decimal:
    with localcontext(Context(prec=DECIMAL_WORK_PRECISION, rounding=ROUND_HALF_EVEN)):
        two_pi = 2 * DECIMAL_PI
        target = mean_anomaly % two_pi
        estimate = target if eccentricity < Decimal("0.8") else DECIMAL_PI
        threshold = Decimal(1).scaleb(-(DECIMAL_PRECISION + 5))
        for _ in range(128):
            sine, cosine = _decimal_sin_cos(estimate)
            delta = (estimate - eccentricity * sine - target) / (1 - eccentricity * cosine)
            estimate -= delta
            if abs(delta) < threshold:
                return +estimate
    raise QualificationError("100-digit Kepler solve did not converge")


@dataclass(frozen=True)
class AnalyticCase:
    semimajor_axis: float
    eccentricity: float
    true_anomaly: float
    ratio: float
    direction: int

    @property
    def case_id(self) -> str:
        payload = [self.semimajor_axis.hex(), self.eccentricity.hex(), self.true_anomaly.hex(), self.ratio.hex(), self.direction]
        return "two-body-" + _domain_sha256("jx.wh-v9.two-body-case.v1", payload)[:20]

    @property
    def step_divisor(self) -> int:
        minimum = max(128.0, 64.0 / ((1.0 - self.eccentricity) ** 1.5))
        return 1 << math.ceil(math.log2(minimum))


def _analytic_cases(profile: str, protocol: dict[str, Any]) -> tuple[AnalyticCase, ...]:
    spec = protocol["workloads"]["analytic_two_body_portfolio"]
    cases = tuple(
        AnalyticCase(float(a), float(e), float(f), float(ratio), int(direction))
        for a in spec["semimajor_axis_au_like"]
        for e in spec["eccentricity"]
        for f in spec["true_anomaly_radians"]
        for ratio in spec["secondary_to_primary_gm_ratio"]
        for direction in spec["direction"]
    )
    if len(cases) != 240:
        raise QualificationError("analytic portfolio cardinality changed")
    return cases if profile == FORMAL else (cases[25],)


def _analytic_state(case: AnalyticCase, epoch: float) -> tuple[np.ndarray, np.ndarray]:
    with localcontext(Context(prec=DECIMAL_WORK_PRECISION, rounding=ROUND_HALF_EVEN)):
        a = Decimal.from_float(case.semimajor_axis)
        e = Decimal.from_float(case.eccentricity)
        k0 = Decimal(1)
        k1 = Decimal.from_float(case.ratio)
        total = k0 + k1
        f = Decimal.from_float(case.true_anomaly)
        sin_f, cos_f = _decimal_sin_cos(f)
        root = (Decimal(1) - e * e).sqrt()
        cos_e0 = (e + cos_f) / (Decimal(1) + e * cos_f)
        sin_e0 = root * sin_f / (Decimal(1) + e * cos_f)
        eccentric0 = _decimal_atan2(sin_e0, cos_e0)
        mean0 = eccentric0 - e * sin_e0
        mean_motion = (total / (a * a * a)).sqrt()
        eccentric = _solve_decimal_kepler(mean0 + mean_motion * Decimal.from_float(epoch), e)
        sine, cosine = _decimal_sin_cos(eccentric)
        denominator = Decimal(1) - e * cosine
        x = a * (cosine - e)
        y = Decimal(case.direction) * a * root * sine
        vx = -a * mean_motion * sine / denominator
        vy = Decimal(case.direction) * a * mean_motion * root * cosine / denominator
        q = (x, y, Decimal(0))
        u = (vx, vy, Decimal(0))
        positions = np.array(
            [[float(-k1 * value / total) for value in q], [float(k0 * value / total) for value in q]], dtype=np.float64
        )
        velocities = np.array(
            [[float(-k1 * value / total) for value in u], [float(k0 * value / total) for value in u]], dtype=np.float64
        )
    return positions, velocities


def _plan_and_snapshot(
    body_ids: tuple[str, ...],
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    *,
    epoch: float,
    validity_start: float,
    validity_end: float,
    fixture_id: str,
) -> tuple[StateSnapshot, ForcePlan]:
    source = _sha256(Path(__file__).read_bytes())
    provenance = Provenance(
        "jx.wh-v9.qualification.runner.v1",
        "benchmarks/jx_wh_v9_scientific_qualification.py",
        "1",
        source,
    )
    snapshot = StateSnapshot(
        f"{fixture_id}.snapshot.v1",
        float(epoch),
        "SYNTHETIC",
        "BARYCENTRIC_INERTIAL",
        "BARYCENTER",
        "ICRS_ALIGNED",
        "L",
        "T",
        "M",
        fixture_id,
        body_ids,
        np.array(positions, dtype=np.float64, copy=True),
        np.array(velocities, dtype=np.float64, copy=True),
        np.array(gm, dtype=np.float64, copy=True),
        np.array(gm, dtype=np.float64, copy=True),
        np.zeros(len(body_ids), dtype=np.float64),
        np.ones(len(body_ids), dtype=np.bool_),
        provenance,
    )
    metadata = ParameterMetadata(
        "state.gravitational_parameters",
        "L^3/T^2",
        provenance,
        None,
        None,
        float(validity_start),
        float(validity_end),
    )
    plan = ForcePlan(
        f"{fixture_id}.plan.v1",
        BackendSpec("numpy", "cpu", 2),
        (NewtonianPointMass(body_ids, body_ids, fixture_id, (metadata,)),),
    )
    return snapshot, plan


def _two_body_inputs(
    case: AnalyticCase, checkpoints: tuple[int, ...]
) -> tuple[StateSnapshot, ForcePlan, FixedStepWisdomHolmanSpec, float]:
    positions, velocities = _analytic_state(case, 0.0)
    total = 1.0 + case.ratio
    period = 2.0 * math.pi * math.sqrt(case.semimajor_axis**3 / total)
    divisor = case.step_divisor
    maximum_period = checkpoints[-1]
    step_indices = tuple(value * divisor for value in checkpoints)
    duration = float((period / divisor) * step_indices[-1])
    snapshot, plan = _plan_and_snapshot(
        ("PRIMARY", "SECONDARY"),
        positions,
        velocities,
        np.array((1.0, case.ratio), dtype=np.float64),
        epoch=0.0,
        validity_start=-period,
        validity_end=duration + period,
        fixture_id="jx.wh-v9.qualification.analytic-two-body.v1",
    )
    periapse = case.semimajor_axis * (1.0 - case.eccentricity)
    speed_scale = math.sqrt(total / case.semimajor_axis)
    spec = FixedStepWisdomHolmanSpec(
        checkpoint_step_indices=step_indices,
        fixed_step=float(period / divisor),
        maximum_steps=maximum_period * divisor,
        jacobi_body_order=("PRIMARY", "SECONDARY"),
        minimum_encounter_pair_separation=float(periapse * 1.0e-5),
        minimum_jacobi_periapse=float(periapse * 0.5),
        maximum_initial_barycenter_position_norm=float(max(case.semimajor_axis, 1.0e-300)),
        maximum_initial_barycenter_velocity_norm=float(max(speed_scale, 1.0e-300)),
    )
    return snapshot, plan, spec, period


def _pairs_float_hex(values: Iterable[tuple[str, Any]]) -> list[list[Any]]:
    return [[name, float(value).hex()] for name, value in values]


def _pairs_json(values: Iterable[tuple[str, Any]]) -> list[list[Any]]:
    return [[name, value] for name, value in values]


def _v9_receipt(result: v9.FastWisdomHolmanV9LoopResult) -> dict[str, Any]:
    return {
        "schema": result.schema,
        "api_id": result.api_id,
        "result_type_id": result.result_type_id,
        "kernel_id": result.kernel_id,
        "scientific_claim_state": result.scientific_claim_state,
        "support_status": result.support_status,
        "public_default": result.public_default,
        "claim_controls": {
            "registry_authorized": result.registry_authorized,
            "qualification_authorized": result.qualification_authorized,
            "production_authorized": result.production_authorized,
            "superiority_claimed": result.superiority_claimed,
        },
        "arrays": {
            "positions": _array_evidence(result.positions),
            "velocities": _array_evidence(result.velocities),
            "checkpoint_epochs": _array_evidence(result.checkpoint_epochs),
        },
        "metrics_hex": _pairs_float_hex(result.metrics),
        "primary_counters": _pairs_json(result.primary_counters),
        "replay_counters": _pairs_json(result.replay_counters),
        "total_counters": _pairs_json(result.total_counters),
        "primary_fallback_count": result.primary_fallback_count,
        "replay_fallback_count": result.replay_fallback_count,
        "total_fallback_count": result.total_fallback_count,
        "primary_cache_stats": _pairs_json(result.primary_cache_stats),
        "replay_cache_stats": _pairs_json(result.replay_cache_stats),
        "primary_fp_environment_before": _pairs_json(result.primary_fp_environment_before),
        "primary_fp_environment_after": _pairs_json(result.primary_fp_environment_after),
        "replay_fp_environment_before": None if result.replay_fp_environment_before is None else _pairs_json(result.replay_fp_environment_before),
        "replay_fp_environment_after": None if result.replay_fp_environment_after is None else _pairs_json(result.replay_fp_environment_after),
        "replay_count": result.replay_count,
        "verification_mode": result.verification_mode,
        "verification_scope": result.verification_scope,
        "certified_steps": result.certified_steps,
        "certified_kepler_solves": result.certified_kepler_solves,
        "full_step_postcondition_coverage": result.full_step_postcondition_coverage,
        "exact_replay_verified": result.exact_replay_verified,
        "independent_numerical_method_verified": result.independent_numerical_method_verified,
        "deterministic_implementation_defect_excluded": result.deterministic_implementation_defect_excluded,
        "verification_content_sha256": result.verification_content_sha256,
        "result_content_sha256": result.result_content_sha256,
        "input_content_sha256": result.input_content_sha256,
        "runtime_identity_sha256": result.runtime_identity_sha256,
        "source_closure_sha256": result.source_closure_sha256,
    }


def _tuple_pairs(value: Any, label: str, *, floats: bool = False) -> tuple[tuple[str, Any], ...]:
    if type(value) is not list or any(type(item) is not list or len(item) != 2 or type(item[0]) is not str for item in value):
        raise QualificationError(f"{label} pair schema changed")
    result: list[tuple[str, Any]] = []
    for name, raw in value:
        parsed = float.fromhex(raw) if floats else raw
        result.append((name, parsed))
    return tuple(result)


def _validate_v9_receipt(receipt: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, v9.FastWisdomHolmanV9LoopResult]:
    required = {
        "schema", "api_id", "result_type_id", "kernel_id", "scientific_claim_state", "support_status", "public_default", "claim_controls",
        "arrays", "metrics_hex", "primary_counters", "replay_counters", "total_counters", "primary_fallback_count", "replay_fallback_count",
        "total_fallback_count", "primary_cache_stats", "replay_cache_stats", "primary_fp_environment_before", "primary_fp_environment_after",
        "replay_fp_environment_before", "replay_fp_environment_after", "replay_count", "verification_mode", "verification_scope", "certified_steps",
        "certified_kepler_solves", "full_step_postcondition_coverage", "exact_replay_verified", "independent_numerical_method_verified",
        "deterministic_implementation_defect_excluded", "verification_content_sha256", "result_content_sha256", "input_content_sha256",
        "runtime_identity_sha256", "source_closure_sha256",
    }
    if type(receipt) is not dict or set(receipt) != required:
        raise QualificationError("V9 receipt schema changed")
    arrays = receipt["arrays"]
    if type(arrays) is not dict or set(arrays) != {"positions", "velocities", "checkpoint_epochs"}:
        raise QualificationError("V9 state-evidence schema changed")
    positions = _decode_array(arrays["positions"])
    velocities = _decode_array(arrays["velocities"], shape=positions.shape)
    epochs = _decode_array(arrays["checkpoint_epochs"], shape=(positions.shape[0],))
    claim = receipt["claim_controls"]
    if claim != {"registry_authorized": False, "qualification_authorized": False, "production_authorized": False, "superiority_claimed": False}:
        raise QualificationError("V9 receipt claim boundary changed")
    before = _tuple_pairs(receipt["primary_fp_environment_before"], "primary fp before")
    after = _tuple_pairs(receipt["primary_fp_environment_after"], "primary fp after")
    replay_before = None if receipt["replay_fp_environment_before"] is None else _tuple_pairs(receipt["replay_fp_environment_before"], "replay fp before")
    replay_after = None if receipt["replay_fp_environment_after"] is None else _tuple_pairs(receipt["replay_fp_environment_after"], "replay fp after")
    rebuilt = v9.FastWisdomHolmanV9LoopResult(
        positions=positions,
        velocities=velocities,
        checkpoint_epochs=epochs,
        metrics=_tuple_pairs(receipt["metrics_hex"], "metrics", floats=True),
        primary_counters=_tuple_pairs(receipt["primary_counters"], "primary counters"),
        replay_counters=_tuple_pairs(receipt["replay_counters"], "replay counters"),
        total_counters=_tuple_pairs(receipt["total_counters"], "total counters"),
        primary_fallback_count=receipt["primary_fallback_count"],
        replay_fallback_count=receipt["replay_fallback_count"],
        total_fallback_count=receipt["total_fallback_count"],
        primary_cache_stats=_tuple_pairs(receipt["primary_cache_stats"], "primary cache"),
        replay_cache_stats=_tuple_pairs(receipt["replay_cache_stats"], "replay cache"),
        primary_fp_environment_before=before,
        primary_fp_environment_after=after,
        replay_fp_environment_before=replay_before,
        replay_fp_environment_after=replay_after,
        primary_seconds=0.0,
        replay_seconds=0.0,
        replay_count=receipt["replay_count"],
        verification_mode=receipt["verification_mode"],
        verification_scope=receipt["verification_scope"],
        certified_steps=receipt["certified_steps"],
        certified_kepler_solves=receipt["certified_kepler_solves"],
        full_step_postcondition_coverage=receipt["full_step_postcondition_coverage"],
        exact_replay_verified=receipt["exact_replay_verified"],
        independent_numerical_method_verified=receipt["independent_numerical_method_verified"],
        deterministic_implementation_defect_excluded=receipt["deterministic_implementation_defect_excluded"],
        verification_content_sha256=receipt["verification_content_sha256"],
        result_content_sha256=receipt["result_content_sha256"],
        input_content_sha256=receipt["input_content_sha256"],
        runtime_identity_sha256=receipt["runtime_identity_sha256"],
        source_closure_sha256=receipt["source_closure_sha256"],
    )
    if (
        receipt["schema"] != rebuilt.schema
        or receipt["api_id"] != rebuilt.api_id
        or receipt["result_type_id"] != rebuilt.result_type_id
        or receipt["kernel_id"] != rebuilt.kernel_id
        or receipt["scientific_claim_state"] != "SCREENING_ONLY"
        or receipt["support_status"] != rebuilt.support_status
        or receipt["public_default"] is not False
        or receipt["replay_count"] != 1
        or receipt["exact_replay_verified"] is not True
    ):
        raise QualificationError("V9 receipt identity or replay requirement changed")
    return positions, velocities, epochs, rebuilt


def _bind_v9_receipt_to_request(
    receipt: Any,
    snapshot: StateSnapshot,
    spec: FixedStepWisdomHolmanSpec,
    runtime_identity: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, v9.FastWisdomHolmanV9LoopResult]:
    """Close a retained V9 certificate over its reconstructed public request."""

    positions, velocities, epochs, result = _validate_v9_receipt(receipt)
    checkpoint_steps = np.asarray(spec.checkpoint_step_indices, dtype=np.int64)
    expected_epochs = np.asarray(
        [float(snapshot.epoch + spec.fixed_step * int(index)) for index in checkpoint_steps],
        dtype=np.float64,
    )
    expected_input_sha256 = v9._v3._workspace_hash(
        body_ids=snapshot.body_ids,
        initial_epoch=float(snapshot.epoch),
        fixed_step=spec.fixed_step,
        checkpoint_steps=checkpoint_steps,
        positions=np.asarray(snapshot.positions, dtype=np.float64),
        velocities=np.asarray(snapshot.velocities, dtype=np.float64),
        gravitational_parameters=np.asarray(
            snapshot.gravitational_parameters, dtype=np.float64
        ),
        radii=np.asarray(snapshot.radii, dtype=np.float64),
        guard_values=(
            spec.minimum_encounter_pair_separation,
            spec.minimum_jacobi_periapse,
            spec.maximum_initial_barycenter_position_norm,
            spec.maximum_initial_barycenter_velocity_norm,
        ),
    )
    if (
        positions.shape
        != (len(spec.checkpoint_step_indices), len(snapshot.body_ids), 3)
        or not _array_bits_equal(epochs, expected_epochs)
        or not _array_bits_equal(
            np.ascontiguousarray(positions[0]),
            np.ascontiguousarray(snapshot.positions, dtype=np.float64),
        )
        or not _array_bits_equal(
            np.ascontiguousarray(velocities[0]),
            np.ascontiguousarray(snapshot.velocities, dtype=np.float64),
        )
        or result.input_content_sha256 != expected_input_sha256
        or result.certified_steps != spec.completed_steps
        or result.runtime_identity_sha256 != runtime_identity["content_sha256"]
        or result.source_closure_sha256
        != runtime_identity["source_closure_sha256"]
    ):
        raise QualificationError(
            "V9 receipt is not bound to the reconstructed input/spec/runtime"
        )
    return positions, velocities, epochs, result


def _two_body_invariants(positions: np.ndarray, velocities: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    energy: list[float] = []
    angular: list[np.ndarray] = []
    gm = (1.0, ratio)
    for p, u in zip(positions, velocities, strict=True):
        value = sum(0.5 * gm[i] * float(np.dot(u[i], u[i])) for i in range(2))
        value -= ratio / float(np.linalg.norm(p[1] - p[0]))
        energy.append(value)
        angular.append(sum((gm[i] * np.cross(p[i], u[i]) for i in range(2)), start=np.zeros(3)))
    return np.asarray(energy), np.asarray(angular)


def _two_body_metrics(
    case: AnalyticCase,
    positions: np.ndarray,
    velocities: np.ndarray,
    oracle_positions: np.ndarray,
    oracle_velocities: np.ndarray,
    backward_positions: np.ndarray,
    backward_velocities: np.ndarray,
) -> dict[str, float]:
    candidate_q = positions[:, 1] - positions[:, 0]
    candidate_u = velocities[:, 1] - velocities[:, 0]
    oracle_q = oracle_positions[:, 1] - oracle_positions[:, 0]
    oracle_u = oracle_velocities[:, 1] - oracle_velocities[:, 0]
    speed = math.sqrt((1.0 + case.ratio) / case.semimajor_axis)
    position_error = float(np.max(np.linalg.norm(candidate_q - oracle_q, axis=1)) / case.semimajor_axis)
    velocity_error = float(np.max(np.linalg.norm(candidate_u - oracle_u, axis=1)) / speed)
    energy, angular = _two_body_invariants(positions, velocities, case.ratio)
    oracle_energy, oracle_angular = _two_body_invariants(oracle_positions, oracle_velocities, case.ratio)
    energy_error = float(np.max(np.abs((energy - oracle_energy) / oracle_energy)))
    angular_scale = np.linalg.norm(oracle_angular, axis=1)
    angular_error = float(np.max(np.linalg.norm(angular - oracle_angular, axis=1) / angular_scale))
    initial_q = positions[0, 1] - positions[0, 0]
    initial_u = velocities[0, 1] - velocities[0, 0]
    backward_q = backward_positions[-1, 1] - backward_positions[-1, 0]
    backward_u = backward_velocities[-1, 1] - backward_velocities[-1, 0]
    roundtrip = max(float(np.linalg.norm(backward_q - initial_q) / case.semimajor_axis), float(np.linalg.norm(backward_u - initial_u) / speed))
    return {
        "normalized_position_error_max": position_error,
        "normalized_velocity_error_max": velocity_error,
        "relative_energy_error_max": energy_error,
        "relative_angular_momentum_error_max": angular_error,
        "normalized_forward_backward_state_error_max": roundtrip,
    }


def _run_analytic_case(case: AnalyticCase, checkpoints: tuple[int, ...]) -> dict[str, Any]:
    snapshot, plan, spec, period = _two_body_inputs(case, checkpoints)
    forward = v9.integrate_fast_wisdom_holman_v9_trajectory(
        snapshot, plan, spec, exact_replay=True
    )
    oracle_positions: list[np.ndarray] = []
    oracle_velocities: list[np.ndarray] = []
    for epoch in forward.checkpoint_epochs:
        position, velocity = _analytic_state(case, float(epoch))
        oracle_positions.append(position)
        oracle_velocities.append(velocity)
    oracle_p = np.stack(oracle_positions)
    oracle_v = np.stack(oracle_velocities)

    endpoint_epoch = float(forward.checkpoint_epochs[-1])
    endpoint_snapshot, reverse_plan = _plan_and_snapshot(
        ("PRIMARY", "SECONDARY"),
        forward.positions[-1],
        forward.velocities[-1],
        np.array((1.0, case.ratio), dtype=np.float64),
        epoch=endpoint_epoch,
        validity_start=-period,
        validity_end=endpoint_epoch + period,
        fixture_id="jx.wh-v9.qualification.analytic-two-body.reverse.v1",
    )
    reverse_steps = spec.checkpoint_step_indices[-1]
    reverse_spec = FixedStepWisdomHolmanSpec(
        checkpoint_step_indices=(0, reverse_steps),
        fixed_step=-spec.fixed_step,
        maximum_steps=reverse_steps,
        jacobi_body_order=("PRIMARY", "SECONDARY"),
        minimum_encounter_pair_separation=spec.minimum_encounter_pair_separation,
        minimum_jacobi_periapse=spec.minimum_jacobi_periapse,
        maximum_initial_barycenter_position_norm=spec.maximum_initial_barycenter_position_norm,
        maximum_initial_barycenter_velocity_norm=spec.maximum_initial_barycenter_velocity_norm,
    )
    backward = v9.integrate_fast_wisdom_holman_v9_trajectory(
        endpoint_snapshot, reverse_plan, reverse_spec, exact_replay=True
    )
    metrics = _two_body_metrics(
        case,
        forward.positions,
        forward.velocities,
        oracle_p,
        oracle_v,
        backward.positions,
        backward.velocities,
    )
    return {
        "case_id": case.case_id,
        "elements": {
            "semimajor_axis_hex": case.semimajor_axis.hex(),
            "eccentricity_hex": case.eccentricity.hex(),
            "true_anomaly_hex": case.true_anomaly.hex(),
            "secondary_to_primary_gm_ratio_hex": case.ratio.hex(),
            "direction": case.direction,
        },
        "step_divisor": case.step_divisor,
        "nominal_period_hex": period.hex(),
        "checkpoint_periods": list(checkpoints),
        "candidate": _v9_receipt(forward),
        "analytic_oracle": {
            "implementation": "STANDALONE_DECIMAL_100_DIGIT_ELLIPTIC_KEPLER_SOLUTION",
            "precision_decimal_digits": DECIMAL_PRECISION,
            "shares_jx_integrator_code": False,
            "positions": _array_evidence(oracle_p),
            "velocities": _array_evidence(oracle_v),
        },
        "backward_candidate": _v9_receipt(backward),
        "metrics_hex": {key: value.hex() for key, value in metrics.items()},
        "fallback_count": forward.total_fallback_count + backward.total_fallback_count,
    }


def _standalone_newtonian_rhs(_epoch: float, state: np.ndarray, gm: np.ndarray) -> np.ndarray:
    body_count = len(gm)
    positions = state[: body_count * 3].reshape(body_count, 3)
    velocities = state[body_count * 3 :].reshape(body_count, 3)
    accelerations = np.zeros((body_count, 3), dtype=np.float64)
    for left in range(body_count - 1):
        for right in range(left + 1, body_count):
            delta = positions[right] - positions[left]
            distance_squared = float(np.dot(delta, delta))
            inverse_cube = 1.0 / (distance_squared * math.sqrt(distance_squared))
            accelerations[left] += gm[right] * inverse_cube * delta
            accelerations[right] -= gm[left] * inverse_cube * delta
    return np.concatenate((velocities.ravel(), accelerations.ravel()))


def _run_dop853(epochs: np.ndarray, *, require_exact_version: bool) -> dict[str, Any]:
    protocol = _protocol()
    expected_version = protocol["independent_oracles"]["dop853"]["required_scipy_version"]
    if require_exact_version and SCIPY_VERSION != expected_version:
        raise QualificationError(
            f"formal DOP853 requires SciPy {expected_version}, found {SCIPY_VERSION}"
        )
    positions, velocities = fixture._initial_arrays()
    gm = np.array(fixture.GRAVITATIONAL_PARAMETERS, dtype=np.float64, copy=True)
    initial = np.concatenate((positions.ravel(), velocities.ravel()))
    solution = solve_ivp(
        lambda epoch, state: _standalone_newtonian_rhs(epoch, state, gm),
        (float(epochs[0]), float(epochs[-1])),
        initial,
        method="DOP853",
        t_eval=np.asarray(epochs, dtype=np.float64),
        rtol=1.0e-13,
        atol=1.0e-15,
        vectorized=False,
        dense_output=False,
    )
    if not solution.success or solution.y.shape != (18, len(epochs)):
        raise QualificationError(f"standalone DOP853 failed: {solution.message}")
    states = solution.y.T
    candidate_positions = states[:, :9].reshape(len(epochs), 3, 3)
    candidate_velocities = states[:, 9:].reshape(len(epochs), 3, 3)
    if not np.array_equal(np.asarray(solution.t), np.asarray(epochs)):
        raise QualificationError("DOP853 did not retain every requested epoch")
    epoch_evidence = _array_evidence(solution.t)
    receipt = {
        "implementation": "SCIPY_SOLVE_IVP_DOP853_STANDALONE_PAIRWISE_NEWTONIAN_RHS",
        "shares_jx_force_or_integrator_code": False,
        "runtime": _scipy_runtime_custody(),
        "settings": {
            "method": "DOP853",
            "relative_tolerance_hex": float(1.0e-13).hex(),
            "absolute_tolerance_hex": float(1.0e-15).hex(),
            "vectorized": False,
            "dense_output": False,
            "t_span_hex": [float(epochs[0]).hex(), float(epochs[-1]).hex()],
            "requested_epochs_sha256": epoch_evidence["sha256"],
            "gravitational_parameters_hex": [float(value).hex() for value in gm],
            "initial_state_sha256": _array_evidence(initial)["sha256"],
            "successful": bool(solution.success),
            "function_evaluations": int(solution.nfev),
            "jacobian_evaluations": int(solution.njev),
            "lu_decompositions": int(solution.nlu),
            "status": int(solution.status),
            "message": str(solution.message),
        },
        "epochs": epoch_evidence,
        "positions": _array_evidence(candidate_positions),
        "velocities": _array_evidence(candidate_velocities),
    }
    return receipt | {
        "content_sha256": _domain_sha256(DOP853_RECEIPT_DOMAIN, receipt)
    }


def _run_ias15(epochs: np.ndarray) -> dict[str, Any]:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise QualificationError("formal IAS15 requires OMP_NUM_THREADS=1")
    rebound = fixture._load_rebound()
    if getattr(rebound, "__version__", None) != "5.1.1":
        raise QualificationError("formal IAS15 requires REBOUND 5.1.1")
    live_runtime = fixture._rebound_runtime_provenance(rebound)
    fixture._validate_rebound_runtime_provenance(live_runtime)
    lane = fixture._ias15_lane(
        "qualification_rebound_ias15", "qualification", epochs, rebound, live_runtime
    )
    receipt = {
        "implementation": "REBOUND_IAS15_ALL_ACTIVE_NEWTONIAN_POINT_MASSES",
        "shares_jx_force_or_integrator_code": False,
        "runtime": _stable_rebound_runtime(live_runtime),
        "settings": lane.settings,
        "accounting": lane.accounting,
        "epochs": _array_evidence(lane.observed_epochs),
        "positions": _array_evidence(lane.positions),
        "velocities": _array_evidence(lane.velocities),
    }
    return receipt | {
        "content_sha256": _domain_sha256(IAS15_RECEIPT_DOMAIN, receipt)
    }


def _logical_external_identity(
    record: Any, logical_path: str
) -> dict[str, Any]:
    if (
        type(record) is not dict
        or set(record) != {"path", "size_bytes", "sha256"}
        or type(record["size_bytes"]) is not int
        or record["size_bytes"] <= 0
        or not _is_sha256(record["sha256"])
    ):
        raise QualificationError("external runtime file identity changed")
    return {
        "logical_path": logical_path,
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
    }


def _stable_rebound_runtime(runtime: Any) -> dict[str, Any]:
    required = {
        "classification", "authority_authorized", "distribution_name", "distribution_version", "distribution_root",
        "rebound_version", "rebound_githash", "rebound_build", "module_source", "c_library", "python_sources",
        "distribution_metadata", "distribution_license_file", "installed_license_expression",
        "project_description_mentions_v3_or_later", "license_text_inconsistency_observed", "official_repository",
        "official_license_page", "legal_or_redistribution_conclusion_authorized", "rebound_code_vendored",
        "rebound_code_copied", "jx_core_linked_to_rebound", "omp_num_threads",
    }
    if type(runtime) is not dict or set(runtime) != required:
        raise QualificationError("REBOUND runtime provenance schema changed")
    return {
        "schema": "jx.wh-v9.qualification.rebound-ias15-runtime.v1",
        "classification": runtime["classification"],
        "authority_authorized": runtime["authority_authorized"],
        "distribution_name": runtime["distribution_name"],
        "distribution_version": runtime["distribution_version"],
        "rebound_version": runtime["rebound_version"],
        "rebound_githash": runtime["rebound_githash"],
        "rebound_build": runtime["rebound_build"],
        "module_source": _logical_external_identity(
            runtime["module_source"], "rebound/__init__.py"
        ),
        "c_library": _logical_external_identity(
            runtime["c_library"], "rebound/NATIVE_LIBREBOUND_BINARY"
        ),
        "python_sources": runtime["python_sources"],
        "distribution_metadata": _logical_external_identity(
            runtime["distribution_metadata"], "rebound-5.1.1.dist-info/METADATA"
        ),
        "distribution_license_file": _logical_external_identity(
            runtime["distribution_license_file"],
            "rebound-5.1.1.dist-info/licenses/LICENSE",
        ),
        "installed_license_expression": runtime["installed_license_expression"],
        "project_description_mentions_v3_or_later": runtime[
            "project_description_mentions_v3_or_later"
        ],
        "license_text_inconsistency_observed": runtime[
            "license_text_inconsistency_observed"
        ],
        "official_repository": runtime["official_repository"],
        "official_license_page": runtime["official_license_page"],
        "legal_or_redistribution_conclusion_authorized": runtime[
            "legal_or_redistribution_conclusion_authorized"
        ],
        "rebound_code_vendored": runtime["rebound_code_vendored"],
        "rebound_code_copied": runtime["rebound_code_copied"],
        "jx_core_linked_to_rebound": runtime["jx_core_linked_to_rebound"],
        "omp_num_threads": runtime["omp_num_threads"],
    }


def _oracle_arrays(record: Any, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(record) is not dict:
        raise QualificationError(f"{label} oracle receipt changed")
    epochs = _decode_array(record.get("epochs"))
    positions = _decode_array(record.get("positions"), shape=(len(epochs), 3, 3))
    velocities = _decode_array(record.get("velocities"), shape=(len(epochs), 3, 3))
    return epochs, positions, velocities


def _run_weak_v9(periods: int, divisor: int) -> dict[str, Any]:
    profile = fixture.Profile(1, periods, 4)
    snapshot, plan = fixture._state_and_plan(profile)
    spec = fixture._wh_spec(profile, "long", divisor)
    result = v9.integrate_fast_wisdom_holman_v9_trajectory(
        snapshot, plan, spec, exact_replay=True
    )
    return _v9_receipt(result)


def _weak_lane_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    reference_positions: np.ndarray,
    reference_velocities: np.ndarray,
) -> dict[str, Any]:
    return {
        "position_vector_l2_max": fixture._cartesian_metrics(positions, reference_positions)["vector_l2_max"],
        "velocity_vector_l2_max": fixture._cartesian_metrics(velocities, reference_velocities)["vector_l2_max"],
        "phase_error_radians_max": fixture._phase_proxy_metrics(positions, reference_positions)["max_abs_radians"],
        "invariants": fixture._invariant_metrics(positions, velocities),
    }


def _run_weak_horizon(periods: int, divisors: tuple[int, ...], *, require_exact_scipy: bool, include_ias15: bool) -> dict[str, Any]:
    lanes: list[dict[str, Any]] = []
    for divisor in divisors:
        candidate = _run_weak_v9(periods, divisor)
        positions, velocities, epochs, _ = _validate_v9_receipt(candidate)
        dop853 = _run_dop853(epochs, require_exact_version=require_exact_scipy)
        if include_ias15:
            ias15 = _run_ias15(epochs)
            ias_epochs, ias_positions, ias_velocities = _oracle_arrays(ias15, "IAS15")
        else:
            ias15 = None
            ias_epochs = ias_positions = ias_velocities = None
        dop_epochs, dop_positions, dop_velocities = _oracle_arrays(dop853, "DOP853")
        if not np.array_equal(dop_epochs, epochs) or (ias_epochs is not None and not np.array_equal(ias_epochs, epochs)):
            raise QualificationError("weak-three-body oracle clock differs from V9 clock")
        reference_p = ias_positions if ias_positions is not None else dop_positions
        reference_v = ias_velocities if ias_velocities is not None else dop_velocities
        metrics = _weak_lane_metrics(positions, velocities, reference_p, reference_v)
        oracle_difference = None
        if ias_positions is not None:
            oracle_difference = {
                "position_vector_l2_max": fixture._cartesian_metrics(dop_positions, ias_positions)["vector_l2_max"],
                "velocity_vector_l2_max": fixture._cartesian_metrics(dop_velocities, ias_velocities)["vector_l2_max"],
            }
        lanes.append(
            {
                "step_divisor": divisor,
                "candidate": candidate,
                "dop853": dop853,
                "ias15": ias15,
                "metrics_hex": {
                    "position_vector_l2_max": metrics["position_vector_l2_max"].hex(),
                    "velocity_vector_l2_max": metrics["velocity_vector_l2_max"].hex(),
                    "phase_error_radians_max": metrics["phase_error_radians_max"].hex(),
                    "invariants": {key: float(value).hex() for key, value in metrics["invariants"].items()},
                },
                "oracle_difference_hex": None if oracle_difference is None else {key: value.hex() for key, value in oracle_difference.items()},
            }
        )
    return {"periods": periods, "lanes": lanes}


def _run_weak_roundtrip(periods: int, divisor: int) -> dict[str, Any]:
    profile = fixture.Profile(1, periods, 4)
    snapshot, plan = fixture._state_and_plan(profile)
    forward_spec = fixture._wh_spec(profile, "long", divisor)
    forward = v9.integrate_fast_wisdom_holman_v9_trajectory(snapshot, plan, forward_spec, exact_replay=True)
    endpoint, reverse_plan = _plan_and_snapshot(
        fixture.BODY_IDS,
        forward.positions[-1],
        forward.velocities[-1],
        fixture.GRAVITATIONAL_PARAMETERS,
        epoch=float(forward.checkpoint_epochs[-1]),
        validity_start=-fixture.PERIOD,
        validity_end=float(forward.checkpoint_epochs[-1] + fixture.PERIOD),
        fixture_id="jx.wh-v9.qualification.weak-three-body.reverse.v1",
    )
    steps = forward_spec.checkpoint_step_indices[-1]
    reverse_spec = FixedStepWisdomHolmanSpec(
        checkpoint_step_indices=(0, steps),
        fixed_step=-forward_spec.fixed_step,
        maximum_steps=steps,
        jacobi_body_order=fixture.BODY_IDS,
        minimum_encounter_pair_separation=0.01,
        minimum_jacobi_periapse=0.05,
        maximum_initial_barycenter_position_norm=1.0e-12,
        maximum_initial_barycenter_velocity_norm=1.0e-12,
    )
    backward = v9.integrate_fast_wisdom_holman_v9_trajectory(endpoint, reverse_plan, reverse_spec, exact_replay=True)
    position_scale = float(np.max(np.linalg.norm(snapshot.positions, axis=1)))
    velocity_scale = float(np.max(np.linalg.norm(snapshot.velocities, axis=1)))
    state_error = max(
        float(np.max(np.linalg.norm(backward.positions[-1] - snapshot.positions, axis=1)) / position_scale),
        float(np.max(np.linalg.norm(backward.velocities[-1] - snapshot.velocities, axis=1)) / velocity_scale),
    )
    return {"forward": _v9_receipt(forward), "backward": _v9_receipt(backward), "normalized_state_error_hex": state_error.hex()}


def _logical_engineering_command(optimization_level: int) -> list[str]:
    command = ["PYTHON_EXECUTABLE", "-B"]
    if optimization_level == 2:
        command.append("-OO")
    command.extend(("-m", "unittest", "-q", *ENGINEERING_TEST_MODULES))
    return command


def _engineering_environment_record() -> dict[str, str]:
    return {
        "inheritance": "ALLOWLIST_ONLY",
        "LANG": "C",
        "LC_ALL": "C",
        "OMP_NUM_THREADS": "1",
        "PATH": "FIXED_/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONOPTIMIZE": "UNSET",
        "PYTHONPATH": "REPOSITORY_SRC_AND_ROOT",
        "TZ": "UTC",
    }


def _engineering_process_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "OMP_NUM_THREADS": "1",
        "PATH": FORMAL_BUILD_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "TZ": "UTC",
    }


def _bytes_record(data: bytes) -> dict[str, Any]:
    return {
        "size_bytes": len(data),
        "sha256": _sha256(data),
        "data_base64": base64.b64encode(data).decode("ascii"),
    }


def _read_bounded_regular_file(
    path: Path, maximum_bytes: int, label: str
) -> tuple[bytes, tuple[int, int]]:
    """Read one pinned, non-symlink regular file without exceeding a cap."""

    if (
        not isinstance(path, Path)
        or type(maximum_bytes) is not int
        or maximum_bytes <= 0
    ):
        raise QualificationError(f"{label} read contract changed")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(os.fspath(path), flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size <= 0
            or opened.st_size > maximum_bytes
        ):
            raise QualificationError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while total <= maximum_bytes:
            chunk = os.read(
                descriptor, min(1024 * 1024, maximum_bytes + 1 - total)
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > maximum_bytes:
            raise QualificationError(f"{label} exceeds its byte cap")
        retained = os.fstat(descriptor)
        data = b"".join(chunks)
        if (
            (retained.st_dev, retained.st_ino)
            != (opened.st_dev, opened.st_ino)
            or retained.st_size != len(data)
            or not data
        ):
            raise QualificationError(f"{label} changed while being read")
        return data, (opened.st_dev, opened.st_ino)
    except QualificationError:
        raise
    except OSError as exc:
        raise QualificationError(
            f"{label} must be a readable non-symlink regular file"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _decode_bytes_record(value: Any, label: str) -> bytes:
    if type(value) is not dict or set(value) != {
        "size_bytes", "sha256", "data_base64"
    }:
        raise QualificationError(f"{label} byte-record schema changed")
    try:
        data = base64.b64decode(value["data_base64"], validate=True)
    except Exception as exc:
        raise QualificationError(f"{label} byte-record encoding changed") from exc
    if (
        type(value["size_bytes"]) is not int
        or value["size_bytes"] != len(data)
        or len(data) > MAX_ENGINEERING_LOG_BYTES
        or not _is_sha256(value["sha256"])
        or value["sha256"] != _sha256(data)
    ):
        raise QualificationError(f"{label} byte-record digest changed")
    return data


def _normalized_unittest_stream(data: bytes) -> bytes:
    return re.sub(
        rb"(?m)^(Ran [0-9]+ tests? in )[0-9]+(?:\.[0-9]+)?s$",
        rb"\1<ELAPSED>s",
        data,
    )


def _unittest_semantic_evidence(stdout: bytes, stderr: bytes) -> dict[str, Any]:
    combined = stdout + b"\n" + stderr
    try:
        text = combined.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError("engineering unittest output is not UTF-8") from exc
    ran = re.findall(
        r"(?m)^Ran ([0-9]+) tests? in [0-9]+(?:\.[0-9]+)?s$", text
    )
    outcomes = re.findall(r"(?m)^OK(?: \(skipped=([0-9]+)\))?$", text)
    if len(ran) != 1 or int(ran[0]) <= 0 or len(outcomes) != 1:
        raise QualificationError("engineering unittest success summary changed")
    normalized_stdout = _normalized_unittest_stream(stdout)
    normalized_stderr = _normalized_unittest_stream(stderr)
    return {
        "tests_run": int(ran[0]),
        "outcome": "OK",
        "skipped": int(outcomes[0] or 0),
        "normalized_stdout": {
            "size_bytes": len(normalized_stdout),
            "sha256": _sha256(normalized_stdout),
        },
        "normalized_stderr": {
            "size_bytes": len(normalized_stderr),
            "sha256": _sha256(normalized_stderr),
        },
    }


def _stable_engineering_evidence(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "jx.wh-v9.qualification.engineering-semantic-evidence.v1",
        "command": value["command"],
        "optimization_level": value["optimization_level"],
        "working_directory": value["working_directory"],
        "environment": value["environment"],
        "execution_head": value["execution_head"],
        "source_closure_sha256": value["source_closure_sha256"],
        "exit_code": value["exit_code"],
        "python_executable_sha256": value["python_executable_sha256"],
        "unittest": value["semantic_evidence"],
    }


def _validate_stable_engineering_evidence(
    value: Any,
    optimization_level: int,
    source_custody: dict[str, Any],
) -> None:
    expected_keys = {
        "schema", "command", "optimization_level", "working_directory",
        "environment", "execution_head", "source_closure_sha256", "exit_code",
        "python_executable_sha256", "unittest",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value["schema"]
        != "jx.wh-v9.qualification.engineering-semantic-evidence.v1"
        or value["command"] != _logical_engineering_command(optimization_level)
        or value["optimization_level"] != optimization_level
        or value["working_directory"] != "REPOSITORY_ROOT"
        or value["environment"] != _engineering_environment_record()
        or value["execution_head"] != source_custody["execution_head"]
        or value["source_closure_sha256"] != source_custody["closure_sha256"]
        or value["exit_code"] != 0
        or not _is_sha256(value["python_executable_sha256"])
    ):
        raise QualificationError("engineering semantic evidence changed")
    semantic = value["unittest"]
    if (
        type(semantic) is not dict
        or set(semantic)
        != {
            "tests_run", "outcome", "skipped", "normalized_stdout",
            "normalized_stderr",
        }
        or type(semantic["tests_run"]) is not int
        or semantic["tests_run"] <= 0
        or semantic["outcome"] != "OK"
        or type(semantic["skipped"]) is not int
        or semantic["skipped"] < 0
    ):
        raise QualificationError("engineering unittest semantics changed")
    for key in ("normalized_stdout", "normalized_stderr"):
        record = semantic[key]
        if (
            type(record) is not dict
            or set(record) != {"size_bytes", "sha256"}
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or record["size_bytes"] > MAX_ENGINEERING_LOG_BYTES
            or not _is_sha256(record["sha256"])
        ):
            raise QualificationError("normalized unittest evidence changed")


def _execute_engineering_lane(
    optimization_level: int, source_custody: dict[str, Any]
) -> dict[str, Any]:
    if optimization_level not in (0, 2):
        raise QualificationError("engineering optimization level must be zero or two")
    actual = [sys.executable, "-B"]
    if optimization_level == 2:
        actual.append("-OO")
    actual.extend(("-m", "unittest", "-q", *ENGINEERING_TEST_MODULES))
    child_environment = _engineering_process_environment()
    completed = subprocess.run(
        actual,
        cwd=ROOT,
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    core = {
        "schema": "jx.wh-v9.qualification.engineering-command-result.v1",
        "command": _logical_engineering_command(optimization_level),
        "optimization_level": optimization_level,
        "working_directory": "REPOSITORY_ROOT",
        "environment": _engineering_environment_record(),
        "execution_head": source_custody["execution_head"],
        "source_closure_sha256": source_custody["closure_sha256"],
        "exit_code": completed.returncode,
        "stdout": _bytes_record(completed.stdout),
        "stderr": _bytes_record(completed.stderr),
        "semantic_evidence": _unittest_semantic_evidence(
            completed.stdout, completed.stderr
        ),
        "python_executable_sha256": _sha256(Path(sys.executable).resolve().read_bytes()),
    }
    return core | {
        "content_sha256": _domain_sha256(
            "jx.wh-v9.qualification.engineering-command-result.v1", core
        )
    }


def _validate_engineering_log(
    value: Any,
    optimization_level: int,
    source_custody: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "schema", "command", "optimization_level", "working_directory", "environment", "execution_head", "source_closure_sha256", "exit_code", "stdout", "stderr", "semantic_evidence",
        "python_executable_sha256", "content_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise QualificationError("engineering command-result schema changed")
    core = {key: value[key] for key in value if key != "content_sha256"}
    if (
        value["schema"]
        != "jx.wh-v9.qualification.engineering-command-result.v1"
        or value["command"] != _logical_engineering_command(optimization_level)
        or value["optimization_level"] != optimization_level
        or value["working_directory"] != "REPOSITORY_ROOT"
        or value["environment"] != _engineering_environment_record()
        or value["execution_head"] != source_custody["execution_head"]
        or value["source_closure_sha256"] != source_custody["closure_sha256"]
        or value["exit_code"] != 0
        or not _is_sha256(value["python_executable_sha256"])
        or value["content_sha256"]
        != _domain_sha256(
            "jx.wh-v9.qualification.engineering-command-result.v1", core
        )
    ):
        raise QualificationError("engineering command did not complete exactly")
    stdout = _decode_bytes_record(value["stdout"], "engineering stdout")
    stderr = _decode_bytes_record(value["stderr"], "engineering stderr")
    if value["semantic_evidence"] != _unittest_semantic_evidence(stdout, stderr):
        raise QualificationError("engineering unittest semantics changed")
    return value


def _load_engineering_log(
    path: Path,
    optimization_level: int,
    source_custody: dict[str, Any],
) -> dict[str, Any]:
    return _validate_engineering_log(
        _load_json_file(path, maximum_bytes=MAX_ENGINEERING_JSON_BYTES),
        optimization_level,
        source_custody,
    )


def _safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        type(name) is str
        and bool(name)
        and not name.startswith("/")
        and "\\" not in name
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def _distribution_metadata_is_exact(data: bytes) -> bool:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    fields: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line:
            break
        if ": " in line:
            key, value = line.split(": ", 1)
            fields.setdefault(key, []).append(value)
    return fields.get("Name") == ["jxplanetx"] and fields.get("Version") == [
        "0.6.0rc17"
    ]


def _precheck_zip_directory(data: bytes) -> None:
    """Reject oversized/ZIP64 directories before ZipFile allocates entries."""

    signature = b"PK\x05\x06"
    offset = data.rfind(signature, max(0, len(data) - 65_557))
    if offset < 0 or offset + 22 > len(data):
        raise QualificationError("wheel ZIP directory is unavailable")
    disk = int.from_bytes(data[offset + 4 : offset + 6], "little")
    central_disk = int.from_bytes(data[offset + 6 : offset + 8], "little")
    disk_entries = int.from_bytes(data[offset + 8 : offset + 10], "little")
    total_entries = int.from_bytes(data[offset + 10 : offset + 12], "little")
    central_size = int.from_bytes(data[offset + 12 : offset + 16], "little")
    central_offset = int.from_bytes(data[offset + 16 : offset + 20], "little")
    comment_size = int.from_bytes(data[offset + 20 : offset + 22], "little")
    if (
        disk != 0
        or central_disk != 0
        or disk_entries != total_entries
        or total_entries == 0
        or total_entries > MAX_ARCHIVE_MEMBERS
        or total_entries == 0xFFFF
        or central_size > 16 * 1024 * 1024
        or central_offset + central_size != offset
        or offset + 22 + comment_size != len(data)
    ):
        raise QualificationError("wheel ZIP directory exceeds its cap")
    cursor = central_offset
    central_end = central_offset + central_size
    observed_entries = 0
    while cursor < central_end:
        if (
            observed_entries >= MAX_ARCHIVE_MEMBERS
            or cursor + 46 > central_end
            or data[cursor : cursor + 4] != b"PK\x01\x02"
        ):
            raise QualificationError("wheel ZIP central directory changed")
        name_size = int.from_bytes(data[cursor + 28 : cursor + 30], "little")
        extra_size = int.from_bytes(data[cursor + 30 : cursor + 32], "little")
        member_comment_size = int.from_bytes(
            data[cursor + 32 : cursor + 34], "little"
        )
        disk_start = int.from_bytes(data[cursor + 34 : cursor + 36], "little")
        record_size = 46 + name_size + extra_size + member_comment_size
        if disk_start != 0 or record_size <= 46 or cursor + record_size > central_end:
            raise QualificationError("wheel ZIP central record changed")
        observed_entries += 1
        cursor += record_size
    if cursor != central_end or observed_entries != total_entries:
        raise QualificationError("wheel ZIP entry count changed")


class _ExpandedByteCap:
    """Minimal read-only stream that caps bytes returned by a decompressor."""

    def __init__(self, stream: Any, maximum_bytes: int) -> None:
        self._stream = stream
        self._maximum_bytes = maximum_bytes
        self._total = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self._maximum_bytes - self._total
        requested = remaining + 1 if size < 0 else min(size, remaining + 1)
        data = self._stream.read(requested)
        self._total += len(data)
        if self._total > self._maximum_bytes:
            raise QualificationError("archive expansion exceeds its byte cap")
        return data


def _expected_packaged_sources(
    source_custody: dict[str, Any], *, sdist: bool
) -> dict[str, tuple[int, str]]:
    relative_paths = [
        "src/jxplanetx/fast_wisdom_holman_v9.py",
        *(f"src/jxplanetx/{name}" for name in V9_NATIVE_SOURCE_SHA256),
    ]
    if sdist:
        relative_paths.insert(0, "pyproject.toml")
    expected: dict[str, tuple[int, str]] = {}
    for relative in relative_paths:
        record = _source_record(source_custody, relative)
        packaged = relative if sdist else relative.removeprefix("src/")
        expected[packaged] = (record["size_bytes"], record["sha256"])
    return expected


def _validate_wheel_bytes(
    data: bytes,
    filename: str,
    source_custody: dict[str, Any],
    runtime_extension: dict[str, Any],
) -> dict[str, Any]:
    if (
        len(data) > MAX_WHEEL_BYTES
        or not data
        or not filename.startswith("jxplanetx-0.6.0rc17-")
        or not filename.endswith(".whl")
    ):
        raise QualificationError("wheel filename/version changed")
    _precheck_zip_directory(data)
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                not infos
                or len(infos) > MAX_ARCHIVE_MEMBERS
                or sum(info.file_size for info in infos)
                > MAX_ARCHIVE_EXPANDED_BYTES
                or len(names) != len(set(names))
                or any(not _safe_archive_name(name.rstrip("/")) for name in names)
                or any(
                    stat.S_ISLNK(info.external_attr >> 16) for info in infos
                )
                or archive.testzip() is not None
            ):
                raise QualificationError("wheel ZIP structure changed")
            expected = _expected_packaged_sources(source_custody, sdist=False)
            for member, (size, digest) in expected.items():
                if member not in names:
                    raise QualificationError(
                        f"wheel is missing retained source: {member}"
                    )
                raw = archive.read(member)
                if len(raw) != size or _sha256(raw) != digest:
                    raise QualificationError(
                        f"wheel retained source changed: {member}"
                    )
            prefix = "jxplanetx-0.6.0rc17.dist-info/"
            metadata_name = prefix + "METADATA"
            if (
                metadata_name not in names
                or prefix + "WHEEL" not in names
                or prefix + "RECORD" not in names
                or not _distribution_metadata_is_exact(
                    archive.read(metadata_name)
                )
            ):
                raise QualificationError("wheel metadata changed")
            extensions = [
                name
                for name in names
                if name.startswith("jxplanetx/_wisdom_holman_cpu_v9.")
                and name.endswith(".so")
            ]
            extension_raw = (
                b"" if len(extensions) != 1 else archive.read(extensions[0])
            )
            extension_record = {
                "member": extensions[0] if len(extensions) == 1 else "",
                "name": Path(extensions[0]).name if len(extensions) == 1 else "",
                "size_bytes": len(extension_raw),
                "sha256": _sha256(extension_raw),
            }
            if (
                len(extensions) != 1
                or not extension_raw
                or type(runtime_extension) is not dict
                or extension_record["name"] != runtime_extension.get("name")
                or extension_record["size_bytes"]
                != runtime_extension.get("size_bytes")
                or extension_record["sha256"] != runtime_extension.get("sha256")
            ):
                raise QualificationError("wheel V9 extension payload changed")
            return extension_record
    except QualificationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise QualificationError("wheel is not a valid ZIP distribution") from exc


def _validate_sdist_bytes(
    data: bytes, filename: str, source_custody: dict[str, Any]
) -> None:
    expected_filename = "jxplanetx-0.6.0rc17.tar.gz"
    if not data or len(data) > MAX_SDIST_BYTES or filename != expected_filename:
        raise QualificationError("sdist filename/version changed")
    root = "jxplanetx-0.6.0rc17/"
    expected = _expected_packaged_sources(source_custody, sdist=True)
    expected_names = {root + relative: (relative, size, digest) for relative, (size, digest) in expected.items()}
    metadata_name = root + "PKG-INFO"
    retained: set[str] = set()
    metadata_bytes: bytes | None = None
    seen: set[str] = set()
    count = 0
    expanded = 0
    try:
        compressed = gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb")
        capped = _ExpandedByteCap(compressed, MAX_ARCHIVE_EXPANDED_BYTES)
        with compressed, tarfile.open(fileobj=capped, mode="r|") as archive:
            for member in archive:
                count += 1
                if (
                    count > MAX_ARCHIVE_MEMBERS
                    or member.name in seen
                    or not _safe_archive_name(member.name.rstrip("/"))
                    or not (
                        member.name.startswith(root)
                        or (
                            member.name == root.rstrip("/")
                            and member.isdir()
                        )
                    )
                    or not (member.isfile() or member.isdir())
                ):
                    raise QualificationError("sdist tar structure changed")
                seen.add(member.name)
                if member.isfile():
                    expanded += member.size
                    if expanded > MAX_ARCHIVE_EXPANDED_BYTES:
                        raise QualificationError(
                            "sdist expansion exceeds its byte cap"
                        )
                expected_record = expected_names.get(member.name)
                if expected_record is not None:
                    relative, size, digest = expected_record
                    if not member.isfile() or member.size != size:
                        raise QualificationError(
                            f"sdist retained source changed: {relative}"
                        )
                    stream = archive.extractfile(member)
                    raw = None if stream is None else stream.read(size + 1)
                    if raw is None or len(raw) != size or _sha256(raw) != digest:
                        raise QualificationError(
                            f"sdist retained source changed: {relative}"
                        )
                    retained.add(member.name)
                elif member.name == metadata_name:
                    if not member.isfile() or member.size > 1024 * 1024:
                        raise QualificationError("sdist metadata changed")
                    stream = archive.extractfile(member)
                    metadata_bytes = (
                        None
                        if stream is None
                        else stream.read(member.size + 1)
                    )
            if count == 0 or retained != set(expected_names):
                raise QualificationError("sdist retained source roster changed")
            if metadata_bytes is None or not _distribution_metadata_is_exact(
                metadata_bytes
            ):
                raise QualificationError("sdist metadata changed")
    except QualificationError:
        raise
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise QualificationError("sdist is not a valid gzip/tar distribution") from exc


def _artifact_pair(
    first: Path,
    second: Path,
    kind: str,
    source_custody: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    try:
        resolved_first = first.resolve(strict=True)
        resolved_second = second.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            f"{kind} artifacts are unavailable"
        ) from exc
    size_cap = MAX_WHEEL_BYTES if kind == "wheel" else MAX_SDIST_BYTES
    first_bytes, first_identity = _read_bounded_regular_file(
        first, size_cap, f"BUILD_A {kind} artifact"
    )
    second_bytes, second_identity = _read_bounded_regular_file(
        second, size_cap, f"BUILD_B {kind} artifact"
    )
    if (
        first_identity == second_identity
        or resolved_first == resolved_second
        or resolved_first.parent == resolved_second.parent
    ):
        raise QualificationError(
            f"{kind} artifacts must be distinct regular files from independent build directories"
        )
    if kind == "wheel":
        valid_names = first.name.endswith(".whl") and second.name.endswith(".whl")
    else:
        valid_names = first.name.endswith(".tar.gz") and second.name.endswith(".tar.gz")
    if not valid_names or not first_bytes or first_bytes != second_bytes:
        raise QualificationError(f"independent {kind} artifact bytes differ")
    extension_record = None
    if kind == "wheel":
        extension_record = _validate_wheel_bytes(
            first_bytes,
            first.name,
            source_custody,
            runtime_identity["extension_binary"],
        )
        second_extension = _validate_wheel_bytes(
            second_bytes,
            second.name,
            source_custody,
            runtime_identity["extension_binary"],
        )
        if second_extension != extension_record:
            raise QualificationError("wheel V9 extension identities differ")
    else:
        _validate_sdist_bytes(first_bytes, first.name, source_custody)
        _validate_sdist_bytes(second_bytes, second.name, source_custody)
    digest = _sha256(first_bytes)
    result = {
        "kind": kind,
        "build_a": {
            "logical_label": f"BUILD_A/{kind.upper()}",
            "filename": first.name,
            "size_bytes": len(first_bytes),
            "sha256": digest,
        },
        "build_b": {
            "logical_label": f"BUILD_B/{kind.upper()}",
            "filename": second.name,
            "size_bytes": len(second_bytes),
            "sha256": digest,
        },
        "byte_identical": True,
    }
    if extension_record is not None:
        result["science_runtime_extension"] = extension_record
        result["matches_science_runtime"] = True
    return result


def _science_runtime_loading_record(
    wheel_pair: dict[str, Any], source_custody: dict[str, Any]
) -> dict[str, Any]:
    artifact = wheel_pair["build_a"]
    extension = wheel_pair["science_runtime_extension"]
    wrapper = _source_record(
        source_custody, "src/jxplanetx/fast_wisdom_holman_v9.py"
    )
    return {
        "schema": "jx.wh-v9.qualification.wheel-science-runtime.v2",
        "loading_mode": (
            "FRESH_WRAPPER_AND_EXTENSION_MODULES_FROM_VALIDATED_BUILD_A_BYTES"
        ),
        "wheel_logical_label": artifact["logical_label"],
        "wheel_filename": artifact["filename"],
        "wheel_size_bytes": artifact["size_bytes"],
        "wheel_sha256": artifact["sha256"],
        "extension_member": extension["member"],
        "extension_name": extension["name"],
        "extension_size_bytes": extension["size_bytes"],
        "extension_sha256": extension["sha256"],
        "wrapper_member": "jxplanetx/fast_wisdom_holman_v9.py",
        "wrapper_size_bytes": wrapper["size_bytes"],
        "wrapper_sha256": wrapper["sha256"],
    }


@contextmanager
def _formal_science_native_from_wheel(
    wheel_path: Path,
    loading_record: dict[str, Any],
    source_custody: dict[str, Any],
    runtime_custody: dict[str, Any],
) -> Iterator[None]:
    """Route formal V9 calls through fresh wrapper/native wheel-A modules."""

    global v9

    wheel_bytes, _ = _read_bounded_regular_file(
        wheel_path, MAX_WHEEL_BYTES, "formal science wheel A"
    )
    if (
        wheel_path.name != loading_record["wheel_filename"]
        or len(wheel_bytes) != loading_record["wheel_size_bytes"]
        or _sha256(wheel_bytes) != loading_record["wheel_sha256"]
    ):
        raise QualificationError("formal science wheel A identity changed")
    extension = _validate_wheel_bytes(
        wheel_bytes,
        wheel_path.name,
        source_custody,
        runtime_custody["v9"]["extension_binary"],
    )
    if loading_record != _science_runtime_loading_record(
        {
            "build_a": {
                "logical_label": "BUILD_A/WHEEL",
                "filename": wheel_path.name,
                "size_bytes": len(wheel_bytes),
                "sha256": _sha256(wheel_bytes),
            },
            "science_runtime_extension": extension,
        },
        source_custody,
    ):
        raise QualificationError("formal wheel-science loading record changed")
    try:
        with zipfile.ZipFile(io.BytesIO(wheel_bytes), "r") as archive:
            extension_bytes = archive.read(extension["member"])
            wrapper_bytes = archive.read(loading_record["wrapper_member"])
            compiled_sources = {
                name: archive.read(f"jxplanetx/{name}")
                for name in V9_NATIVE_SOURCE_SHA256
            }
    except (OSError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise QualificationError(
            "formal science modules could not be read from wheel A"
        ) from exc
    if (
        len(wrapper_bytes) != loading_record["wrapper_size_bytes"]
        or _sha256(wrapper_bytes) != loading_record["wrapper_sha256"]
    ):
        raise QualificationError("formal science wrapper bytes changed")
    original_wrapper = v9
    native_module_name = (
        "jxplanetx._formal_wheel_runtime._wisdom_holman_cpu_v9"
    )
    wrapper_module_name = (
        "jxplanetx._formal_wheel_fast_wisdom_holman_v9"
    )
    with tempfile.TemporaryDirectory(
        prefix="jx-wh-v9-wheel-runtime-"
    ) as temporary:
        package_directory = Path(temporary) / "jxplanetx"
        package_directory.mkdir(mode=0o700)
        extension_path = package_directory / extension["name"]
        wrapper_path = package_directory / "fast_wisdom_holman_v9.py"
        _write_new_file(extension_path, extension_bytes)
        _write_new_file(wrapper_path, wrapper_bytes)
        for name, raw in compiled_sources.items():
            _write_new_file(package_directory / name, raw)
        spec = importlib.util.spec_from_file_location(
            native_module_name, extension_path
        )
        if spec is None or spec.loader is None:
            raise QualificationError(
                "formal science extension loader is unavailable"
            )
        loaded_wrapper = types.ModuleType(wrapper_module_name)
        loaded_wrapper.__file__ = os.fspath(wrapper_path)
        loaded_wrapper.__package__ = "jxplanetx"
        loaded_wrapper.__spec__ = importlib.util.spec_from_loader(
            wrapper_module_name, loader=None, origin=os.fspath(wrapper_path)
        )
        sys.modules[wrapper_module_name] = loaded_wrapper
        try:
            loaded_native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded_native)
            exec(
                compile(wrapper_bytes, os.fspath(wrapper_path), "exec"),
                loaded_wrapper.__dict__,
            )
            loaded_wrapper._native = loaded_native
            v9 = loaded_wrapper
            if _runtime_custody() != runtime_custody:
                raise QualificationError(
                    "wheel-loaded science runtime identity changed"
                )
            yield
        except QualificationError:
            raise
        except Exception as exc:
            raise QualificationError(
                "formal science modules could not be loaded from wheel A"
            ) from exc
        finally:
            v9 = original_wrapper
            if sys.modules.get(wrapper_module_name) is loaded_wrapper:
                del sys.modules[wrapper_module_name]


def _build_tool_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("setuptools", "wheel", "packaging"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise QualificationError(
                f"formal release tool is not installed: {name}"
            ) from exc
    if result != ENGINEERING_TOOL_VERSIONS:
        raise QualificationError(
            f"formal release tool versions changed: {result!r}"
        )
    return result


def _release_environment_record() -> dict[str, Any]:
    return {
        "inheritance": "ALLOWLIST_ONLY",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONOPTIMIZE": "UNSET",
        "PATH": "FIXED_/usr/bin:/bin",
        "SOURCE_DATE_EPOCH": str(RELEASE_SOURCE_DATE_EPOCH),
        "TZ": "UTC",
    }


def _release_process_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "OMP_NUM_THREADS": "1",
        "PATH": FORMAL_BUILD_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "SOURCE_DATE_EPOCH": str(RELEASE_SOURCE_DATE_EPOCH),
        "TZ": "UTC",
    }


def _build_configuration_custody() -> dict[str, Any]:
    variables = {
        name: sysconfig.get_config_var(name)
        for name in (
            "AR", "ARFLAGS", "CC", "CCSHARED", "CFLAGS", "CXX",
            "LDSHARED",
        )
    }
    if any(type(value) is not str or not value for value in variables.values()):
        raise QualificationError("Python build configuration is incomplete")
    return {
        "schema": "jx.wh-v9.qualification.python-build-config.v1",
        "variables": variables,
        "content_sha256": _domain_sha256(
            "jx.wh-v9.qualification.python-build-config.v1", variables
        ),
    }


def _validate_build_configuration_custody(value: Any) -> None:
    if (
        type(value) is not dict
        or set(value) != {"schema", "variables", "content_sha256"}
        or value["schema"]
        != "jx.wh-v9.qualification.python-build-config.v1"
        or type(value["variables"]) is not dict
        or set(value["variables"])
        != {"AR", "ARFLAGS", "CC", "CCSHARED", "CFLAGS", "CXX", "LDSHARED"}
        or any(
            type(item) is not str or not item
            for item in value["variables"].values()
        )
        or value["content_sha256"]
        != _domain_sha256(
            "jx.wh-v9.qualification.python-build-config.v1",
            value["variables"],
        )
    ):
        raise QualificationError("Python build configuration custody changed")


def _compiler_custody() -> dict[str, Any]:
    driver = "x86_64-linux-gnu-gcc"
    located = shutil.which(driver, path=FORMAL_BUILD_PATH)
    if located is None:
        raise QualificationError("formal GCC driver is unavailable")
    executable = Path(located).resolve(strict=True)
    if executable.is_symlink() or not executable.is_file():
        raise QualificationError("formal GCC driver identity changed")
    completed = subprocess.run(
        (str(executable), "-dumpfullversion", "-dumpversion"),
        cwd=ROOT,
        env={"PATH": FORMAL_BUILD_PATH, "LC_ALL": "C"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    version = completed.stdout.strip()
    raw, _ = _read_bounded_regular_file(
        executable, 64 * 1024 * 1024, "formal GCC driver"
    )
    if (
        completed.returncode != 0
        or completed.stderr != ""
        or version != EXPECTED_COMPILER_VERSION
        or not raw
    ):
        raise QualificationError("formal GCC version changed")
    return {
        "schema": "jx.wh-v9.qualification.compiler-custody.v1",
        "driver_command": driver,
        "resolved_basename": executable.name,
        "version": version,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _validate_compiler_custody(value: Any) -> None:
    if (
        type(value) is not dict
        or set(value)
        != {
            "schema", "driver_command", "resolved_basename", "version",
            "size_bytes", "sha256",
        }
        or value["schema"]
        != "jx.wh-v9.qualification.compiler-custody.v1"
        or value["driver_command"] != "x86_64-linux-gnu-gcc"
        or type(value["resolved_basename"]) is not str
        or "/" in value["resolved_basename"]
        or "\\" in value["resolved_basename"]
        or value["version"] != EXPECTED_COMPILER_VERSION
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] <= 0
        or not _is_sha256(value["sha256"])
    ):
        raise QualificationError("formal compiler custody changed")


def _logical_release_build_command(label: str) -> list[str]:
    if label not in ("BUILD_A", "BUILD_B"):
        raise QualificationError("release-build label changed")
    return [
        "PYTHON_EXECUTABLE",
        "-B",
        "tools/build_release.py",
        "--output-dir",
        f"FORMAL_{label}_OUTPUT",
        "--source-date-epoch",
        str(RELEASE_SOURCE_DATE_EPOCH),
    ]


def _read_release_manifest(path: Path) -> dict[str, Any]:
    data, _ = _read_bounded_regular_file(
        path, 1024 * 1024, "release manifest"
    )
    try:
        value = json.loads(
            data.decode("ascii"), object_pairs_hook=_unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError("release manifest is not strict JSON") from exc
    if type(value) is not dict:
        raise QualificationError("release manifest root changed")
    return value


def _validate_release_manifest(
    value: Any, output_directory: Path | None = None
) -> None:
    if (
        type(value) is not dict
        or set(value)
        != {
            "schema", "version", "scientific_claim_state", "source_date_epoch",
            "build_backend", "independent_clean_builds", "byte_identical",
            "artifacts",
        }
        or value["schema"] != "jxplanetx.reproducible-release-artifacts.v1"
        or value["version"] != "0.6.0rc17"
        or value["scientific_claim_state"] != "SCREENING_ONLY"
        or value["source_date_epoch"] != RELEASE_SOURCE_DATE_EPOCH
        or value["build_backend"] != "setuptools==78.1.1"
        or value["independent_clean_builds"] != 2
        or value["byte_identical"] != {"wheel": True, "sdist": True}
        or type(value["artifacts"]) is not dict
        or set(value["artifacts"]) != {"wheel", "sdist"}
    ):
        raise QualificationError("release-build manifest changed")
    for kind, suffix in (("wheel", ".whl"), ("sdist", ".tar.gz")):
        record = value["artifacts"][kind]
        if (
            type(record) is not dict
            or set(record) != {"filename", "size_bytes", "sha256"}
            or type(record["filename"]) is not str
            or "/" in record["filename"]
            or "\\" in record["filename"]
            or not record["filename"].endswith(suffix)
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
            or record["size_bytes"]
            > (MAX_WHEEL_BYTES if kind == "wheel" else MAX_SDIST_BYTES)
            or not _is_sha256(record["sha256"])
        ):
            raise QualificationError("release-build artifact record changed")
        if output_directory is not None:
            artifact = output_directory / record["filename"]
            raw, _ = _read_bounded_regular_file(
                artifact,
                MAX_WHEEL_BYTES if kind == "wheel" else MAX_SDIST_BYTES,
                f"release-build {kind} artifact",
            )
            if len(raw) != record["size_bytes"] or _sha256(raw) != record["sha256"]:
                raise QualificationError("release-build artifact bytes changed")


def _validate_release_build_execution(
    value: Any, label: str, source_custody: dict[str, Any]
) -> None:
    if (
        type(value) is not dict
        or set(value)
        != {
            "schema", "logical_build", "command", "working_directory",
            "environment", "execution_head", "source_closure_sha256",
            "python_executable_sha256", "exit_code", "stdout_manifest_matches",
            "compiler", "python_build_configuration", "stderr", "manifest",
        }
        or value["schema"]
        != "jx.wh-v9.qualification.release-build-execution.v1"
        or value["logical_build"] != label
        or value["command"] != _logical_release_build_command(label)
        or value["working_directory"] != "REPOSITORY_ROOT"
        or value["environment"] != _release_environment_record()
        or value["execution_head"] != source_custody["execution_head"]
        or value["source_closure_sha256"] != source_custody["closure_sha256"]
        or not _is_sha256(value["python_executable_sha256"])
        or value["exit_code"] != 0
        or value["stdout_manifest_matches"] is not True
    ):
        raise QualificationError("release-build execution record changed")
    _validate_compiler_custody(value["compiler"])
    _validate_build_configuration_custody(
        value["python_build_configuration"]
    )
    if _decode_bytes_record(value["stderr"], "release-build stderr") != b"":
        raise QualificationError("release-build emitted stderr")
    _validate_release_manifest(value["manifest"])


def _execute_release_build(
    label: str,
    output_directory: Path,
    source_custody: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    actual = [
        sys.executable,
        "-B",
        str(ROOT / "tools/build_release.py"),
        "--output-dir",
        str(output_directory),
        "--source-date-epoch",
        str(RELEASE_SOURCE_DATE_EPOCH),
    ]
    environment = _release_process_environment()
    compiler = _compiler_custody()
    build_configuration = _build_configuration_custody()
    completed = subprocess.run(
        actual,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise QualificationError(f"formal {label} release build failed")
    manifest = _read_release_manifest(output_directory / "ARTIFACTS.json")
    _validate_release_manifest(manifest, output_directory)
    try:
        stdout_manifest = json.loads(
            completed.stdout.decode("ascii"), object_pairs_hook=_unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError("release-build stdout changed") from exc
    if stdout_manifest != manifest:
        raise QualificationError("release-build stdout/manifest differ")
    record = {
        "schema": "jx.wh-v9.qualification.release-build-execution.v1",
        "logical_build": label,
        "command": _logical_release_build_command(label),
        "working_directory": "REPOSITORY_ROOT",
        "environment": _release_environment_record(),
        "execution_head": source_custody["execution_head"],
        "source_closure_sha256": source_custody["closure_sha256"],
        "python_executable_sha256": _sha256(
            Path(sys.executable).resolve().read_bytes()
        ),
        "exit_code": completed.returncode,
        "stdout_manifest_matches": True,
        "compiler": compiler,
        "python_build_configuration": build_configuration,
        "stderr": _bytes_record(completed.stderr),
        "manifest": manifest,
    }
    _validate_release_build_execution(record, label, source_custody)
    return record, {
        kind: output_directory / manifest["artifacts"][kind]["filename"]
        for kind in ("wheel", "sdist")
    }


def _formal_build_verification(
    paths: EngineeringEvidencePaths,
    source_custody: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jx-wh-v9-formal-build-") as temporary:
        root = Path(temporary)
        build_a, paths_a = _execute_release_build(
            "BUILD_A", root / "build-a", source_custody
        )
        build_b, paths_b = _execute_release_build(
            "BUILD_B", root / "build-b", source_custody
        )
        if (
            build_a["compiler"] != build_b["compiler"]
            or build_a["python_build_configuration"]
            != build_b["python_build_configuration"]
        ):
            raise QualificationError(
                "formal release builds used different compiler configurations"
            )
        generated_paths = EngineeringEvidencePaths(
            paths.normal_log,
            paths.optimized_log,
            paths_a["wheel"],
            paths_b["wheel"],
            paths_a["sdist"],
            paths_b["sdist"],
        )
        generated = {
            "wheel": _artifact_pair(
                generated_paths.wheel_build_a,
                generated_paths.wheel_build_b,
                "wheel",
                source_custody,
                runtime_identity,
            ),
            "sdist": _artifact_pair(
                generated_paths.sdist_build_a,
                generated_paths.sdist_build_b,
                "sdist",
                source_custody,
                runtime_identity,
            ),
        }
        supplied = {
            "wheel": _artifact_pair(
                paths.wheel_build_a,
                paths.wheel_build_b,
                "wheel",
                source_custody,
                runtime_identity,
            ),
            "sdist": _artifact_pair(
                paths.sdist_build_a,
                paths.sdist_build_b,
                "sdist",
                source_custody,
                runtime_identity,
            ),
        }
        if generated != supplied:
            raise QualificationError(
                "supplied artifacts do not match two formal clean release builds"
            )
    return {
        "schema": "jx.wh-v9.qualification.formal-build-verification.v1",
        "build_a": build_a,
        "build_b": build_b,
        "artifacts": generated,
        "generated_artifacts_match_supplied": True,
    }


def _derive_engineering_receipt(
    paths: EngineeringEvidencePaths,
    protocol: dict[str, Any],
    source_custody: dict[str, Any],
    runtime_custody: dict[str, Any],
    *,
    formal_verify: bool,
) -> dict[str, Any]:
    if type(paths) is not EngineeringEvidencePaths:
        raise QualificationError("formal engineering evidence paths are required")
    normal = _load_engineering_log(paths.normal_log, 0, source_custody)
    optimized = _load_engineering_log(paths.optimized_log, 2, source_custody)
    formal_verification = None
    if formal_verify:
        rerun_normal = _execute_engineering_lane(0, source_custody)
        rerun_optimized = _execute_engineering_lane(2, source_custody)
        if (
            _stable_engineering_evidence(normal)
            != _stable_engineering_evidence(rerun_normal)
            or _stable_engineering_evidence(optimized)
            != _stable_engineering_evidence(rerun_optimized)
        ):
            raise QualificationError(
                "retained engineering logs differ from formal test reruns"
            )
        release_builds = _formal_build_verification(
            paths, source_custody, runtime_custody["v9"]
        )
        formal_verification = {
            "schema": "jx.wh-v9.qualification.formal-execution-verification.v1",
            "test_reexecution": {
                "normal_python": _stable_engineering_evidence(rerun_normal),
                "optimized_python": _stable_engineering_evidence(
                    rerun_optimized
                ),
            },
            "release_builds": release_builds,
            "science_runtime_loading": _science_runtime_loading_record(
                release_builds["artifacts"]["wheel"], source_custody
            ),
        }
    return _assemble_engineering_receipt(
        normal,
        optimized,
        paths,
        protocol,
        source_custody,
        runtime_custody,
        formal_verification=formal_verification,
    )


def _assemble_engineering_receipt(
    normal: dict[str, Any],
    optimized: dict[str, Any],
    paths: EngineeringEvidencePaths,
    protocol: dict[str, Any],
    source_custody: dict[str, Any],
    runtime_custody: dict[str, Any],
    *,
    formal_verification: dict[str, Any] | None,
) -> dict[str, Any]:
    _validate_engineering_log(normal, 0, source_custody)
    _validate_engineering_log(optimized, 2, source_custody)
    if normal["python_executable_sha256"] != optimized["python_executable_sha256"]:
        raise QualificationError("normal and optimized logs used different Python binaries")
    if (
        normal["python_executable_sha256"]
        != runtime_custody["python"]["executable_sha256"]
    ):
        raise QualificationError("engineering/science Python binaries differ")
    core = {
        "schema": ENGINEERING_SCHEMA,
        "candidate_base_commit": protocol["base_repository_commit"],
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_head": source_custody["execution_head"],
        "source_closure_sha256": source_custody["closure_sha256"],
        "evidence_ingestion": "PATH_BACKED_BYTES_REVALIDATED_AT_FORMAL_EXECUTION",
        "checks": {
            key: True for key in protocol["gates"]["engineering_prerequisites"]
        },
        "normal_python": _stable_engineering_evidence(normal),
        "optimized_python": _stable_engineering_evidence(optimized),
        "artifacts": {
            "wheel": _artifact_pair(
                paths.wheel_build_a,
                paths.wheel_build_b,
                "wheel",
                source_custody,
                runtime_custody["v9"],
            ),
            "sdist": _artifact_pair(
                paths.sdist_build_a,
                paths.sdist_build_b,
                "sdist",
                source_custody,
                runtime_custody["v9"],
            ),
        },
        "build_tool_versions": _build_tool_versions(),
        "formal_execution_verification": formal_verification,
    }
    receipt = core | {
        "content_sha256": _domain_sha256(ENGINEERING_SCHEMA, core)
    }
    return _engineering_receipt(
        receipt, protocol, source_custody, runtime_custody
    )  # type: ignore[return-value]


def create_engineering_receipt(
    paths: EngineeringEvidencePaths,
    receipt_output: Path,
) -> dict[str, Any]:
    """Run fixed test lanes and create path-backed logs plus a closed receipt.

    This mode never runs a scientific workload.  The receipt remains ancillary:
    formal qualification re-opens the two logs and four build artifacts by path.
    """

    if type(paths) is not EngineeringEvidencePaths or not isinstance(
        receipt_output, Path
    ):
        raise QualificationError("engineering receipt creation paths changed")
    outputs = (paths.normal_log, paths.optimized_log, receipt_output)
    lexical_outputs = tuple(os.path.abspath(os.fspath(path)) for path in outputs)
    if len(set(lexical_outputs)) != len(lexical_outputs):
        raise QualificationError("engineering output paths must be distinct")
    for path in outputs:
        if os.path.lexists(os.fspath(path)):
            raise QualificationError("engineering output path must not exist")
    protocol = _protocol()
    source_custody = _source_custody(require_clean=True)
    runtime_custody = _runtime_custody()
    normal = _execute_engineering_lane(0, source_custody)
    optimized = _execute_engineering_lane(2, source_custody)
    _validate_engineering_log(normal, 0, source_custody)
    _validate_engineering_log(optimized, 2, source_custody)
    receipt = _assemble_engineering_receipt(
        normal,
        optimized,
        paths,
        protocol,
        source_custody,
        runtime_custody,
        formal_verification=None,
    )
    _write_new_file(paths.normal_log, _canonical_bytes(normal, newline=True))
    _write_new_file(
        paths.optimized_log, _canonical_bytes(optimized, newline=True)
    )
    _write_new_file(receipt_output, _canonical_bytes(receipt, newline=True))
    return receipt


def _engineering_receipt(
    value: Any,
    protocol: dict[str, Any],
    source_custody: dict[str, Any],
    runtime_custody: dict[str, Any],
) -> dict[str, Any] | None:
    if value is None:
        return None
    required = {
        "schema", "candidate_base_commit", "protocol_sha256", "execution_head", "source_closure_sha256",
        "evidence_ingestion", "checks", "normal_python", "optimized_python", "artifacts", "build_tool_versions", "formal_execution_verification", "content_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise QualificationError("engineering-receipt schema changed")
    core = {key: value[key] for key in value if key != "content_sha256"}
    expected_checks = set(protocol["gates"]["engineering_prerequisites"])
    if (
        value["schema"] != ENGINEERING_SCHEMA
        or value["candidate_base_commit"] != protocol["base_repository_commit"]
        or value["protocol_sha256"] != PROTOCOL_SHA256
        or value["execution_head"] != source_custody["execution_head"]
        or value["source_closure_sha256"] != source_custody["closure_sha256"]
        or value["evidence_ingestion"]
        != "PATH_BACKED_BYTES_REVALIDATED_AT_FORMAL_EXECUTION"
        or value["content_sha256"] != _domain_sha256(ENGINEERING_SCHEMA, core)
        or type(value["checks"]) is not dict
        or set(value["checks"]) != expected_checks
        or any(value["checks"][key] is not True for key in expected_checks)
    ):
        raise QualificationError("engineering-receipt identity/checks changed")
    _validate_stable_engineering_evidence(
        value["normal_python"], 0, source_custody
    )
    _validate_stable_engineering_evidence(
        value["optimized_python"], 2, source_custody
    )
    if (
        value["normal_python"]["python_executable_sha256"]
        != runtime_custody["python"]["executable_sha256"]
        or value["optimized_python"]["python_executable_sha256"]
        != runtime_custody["python"]["executable_sha256"]
    ):
        raise QualificationError("engineering/science Python custody changed")
    tools = value["build_tool_versions"]
    if tools != ENGINEERING_TOOL_VERSIONS:
        raise QualificationError("engineering build-tool versions changed")
    artifacts = value["artifacts"]
    if type(artifacts) is not dict or set(artifacts) != {"wheel", "sdist"}:
        raise QualificationError("engineering artifact roster changed")
    for kind in ("wheel", "sdist"):
        pair = artifacts[kind]
        expected_pair_keys = {"kind", "build_a", "build_b", "byte_identical"}
        if kind == "wheel":
            expected_pair_keys |= {
                "science_runtime_extension", "matches_science_runtime"
            }
        if type(pair) is not dict or set(pair) != expected_pair_keys or pair["kind"] != kind or pair["byte_identical"] is not True:
            raise QualificationError("engineering artifact-pair schema changed")
        for label, logical in (("build_a", f"BUILD_A/{kind.upper()}"), ("build_b", f"BUILD_B/{kind.upper()}")):
            artifact = pair[label]
            if type(artifact) is not dict or set(artifact) != {"logical_label", "filename", "size_bytes", "sha256"} or artifact["logical_label"] != logical or type(artifact["filename"]) is not str or "/" in artifact["filename"] or "\\" in artifact["filename"] or type(artifact["size_bytes"]) is not int or artifact["size_bytes"] <= 0 or not _is_sha256(artifact["sha256"]):
                raise QualificationError("engineering artifact identity changed")
        if pair["build_a"]["size_bytes"] != pair["build_b"]["size_bytes"] or pair["build_a"]["sha256"] != pair["build_b"]["sha256"] or pair["build_a"]["filename"] != pair["build_b"]["filename"]:
            raise QualificationError("engineering independently built artifacts differ")
        if kind == "wheel":
            extension = pair["science_runtime_extension"]
            runtime_extension = runtime_custody["v9"]["extension_binary"]
            if (
                pair["matches_science_runtime"] is not True
                or type(extension) is not dict
                or set(extension) != {"member", "name", "size_bytes", "sha256"}
                or type(extension["member"]) is not str
                or not extension["member"].startswith("jxplanetx/")
                or extension["name"] != runtime_extension["name"]
                or extension["size_bytes"] != runtime_extension["size_bytes"]
                or extension["sha256"] != runtime_extension["sha256"]
            ):
                raise QualificationError(
                    "wheel/science runtime extension binding changed"
                )
    formal = value["formal_execution_verification"]
    if formal is not None:
        if (
            type(formal) is not dict
            or set(formal)
            != {
                "schema", "test_reexecution", "release_builds",
                "science_runtime_loading",
            }
            or formal["schema"]
            != "jx.wh-v9.qualification.formal-execution-verification.v1"
            or type(formal["test_reexecution"]) is not dict
            or set(formal["test_reexecution"])
            != {"normal_python", "optimized_python"}
            or formal["test_reexecution"]["normal_python"]
            != value["normal_python"]
            or formal["test_reexecution"]["optimized_python"]
            != value["optimized_python"]
        ):
            raise QualificationError("formal test reexecution evidence changed")
        release = formal["release_builds"]
        if (
            type(release) is not dict
            or set(release)
            != {
                "schema", "build_a", "build_b", "artifacts",
                "generated_artifacts_match_supplied",
            }
            or release["schema"]
            != "jx.wh-v9.qualification.formal-build-verification.v1"
            or release["generated_artifacts_match_supplied"] is not True
            or release["artifacts"] != artifacts
        ):
            raise QualificationError("formal release-build verification changed")
        if formal["science_runtime_loading"] != _science_runtime_loading_record(
            artifacts["wheel"], source_custody
        ):
            raise QualificationError("formal wheel-science binding changed")
        for label, key in (("BUILD_A", "build_a"), ("BUILD_B", "build_b")):
            execution = release[key]
            _validate_release_build_execution(
                execution, label, source_custody
            )
            if (
                execution["python_executable_sha256"]
                != runtime_custody["python"]["executable_sha256"]
            ):
                raise QualificationError("release/science Python custody changed")
            manifest_artifacts = execution["manifest"]["artifacts"]
            pair_key = "build_a" if label == "BUILD_A" else "build_b"
            for kind in ("wheel", "sdist"):
                retained = artifacts[kind][pair_key]
                manifest = manifest_artifacts[kind]
                if any(
                    retained[name] != manifest[name]
                    for name in ("filename", "size_bytes", "sha256")
                ):
                    raise QualificationError(
                        "release manifest/artifact receipt changed"
                    )
        if (
            release["build_a"]["compiler"]
            != release["build_b"]["compiler"]
            or release["build_a"]["python_build_configuration"]
            != release["build_b"]["python_build_configuration"]
        ):
            raise QualificationError("formal release compiler custody differs")
    return value


def _gate_le(gates: list[dict[str, Any]], gate_id: str, observed: float, threshold: float) -> None:
    gates.append({"gate_id": gate_id, "comparator": "LESS_THAN_OR_EQUAL", "observed_hex": float(observed).hex(), "threshold_hex": float(threshold).hex(), "passed": bool(math.isfinite(observed) and observed <= threshold)})


def _gate_range(gates: list[dict[str, Any]], gate_id: str, observed: float, minimum: float, maximum: float) -> None:
    gates.append({"gate_id": gate_id, "comparator": "CLOSED_INTERVAL", "observed_hex": float(observed).hex(), "minimum_hex": float(minimum).hex(), "maximum_hex": float(maximum).hex(), "passed": bool(math.isfinite(observed) and minimum <= observed <= maximum)})


def _gate_bool(gates: list[dict[str, Any]], gate_id: str, passed: bool) -> None:
    gates.append({"gate_id": gate_id, "comparator": "IS_TRUE", "observed": bool(passed), "passed": bool(passed)})


def _validate_dop853_record(
    record: Any,
    epochs: np.ndarray,
    runtime: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(record) is not dict or set(record) != {
        "implementation", "shares_jx_force_or_integrator_code", "runtime", "settings", "epochs", "positions", "velocities", "content_sha256"
    }:
        raise QualificationError("DOP853 receipt schema changed")
    core = {key: record[key] for key in record if key != "content_sha256"}
    if (
        record["implementation"]
        != "SCIPY_SOLVE_IVP_DOP853_STANDALONE_PAIRWISE_NEWTONIAN_RHS"
        or record["shares_jx_force_or_integrator_code"] is not False
        or record["runtime"] != runtime
        or record["content_sha256"]
        != _domain_sha256(DOP853_RECEIPT_DOMAIN, core)
    ):
        raise QualificationError("DOP853 runtime/implementation changed")
    observed_epochs, positions, velocities = _oracle_arrays(record, "DOP853")
    if not _array_bits_equal(observed_epochs, epochs):
        raise QualificationError("DOP853 clock evidence changed")
    initial_positions, initial_velocities = fixture._initial_arrays()
    initial_state = np.concatenate((initial_positions.ravel(), initial_velocities.ravel()))
    settings = record["settings"]
    expected_fixed = {
        "method": "DOP853",
        "relative_tolerance_hex": float(1.0e-13).hex(),
        "absolute_tolerance_hex": float(1.0e-15).hex(),
        "vectorized": False,
        "dense_output": False,
        "t_span_hex": [float(epochs[0]).hex(), float(epochs[-1]).hex()],
        "requested_epochs_sha256": record["epochs"]["sha256"],
        "gravitational_parameters_hex": [
            float(value).hex() for value in fixture.GRAVITATIONAL_PARAMETERS
        ],
        "initial_state_sha256": _array_evidence(initial_state)["sha256"],
        "successful": True,
        "jacobian_evaluations": 0,
        "lu_decompositions": 0,
        "status": 0,
        "message": "The solver successfully reached the end of the integration interval.",
    }
    if (
        type(settings) is not dict
        or set(settings) != set(expected_fixed) | {"function_evaluations"}
        or any(settings[key] != value for key, value in expected_fixed.items())
        or type(settings["function_evaluations"]) is not int
        or settings["function_evaluations"] <= 0
        or not _array_bits_equal(
            np.ascontiguousarray(positions[0]), initial_positions
        )
        or not _array_bits_equal(
            np.ascontiguousarray(velocities[0]), initial_velocities
        )
    ):
        raise QualificationError("DOP853 settings/accounting/input changed")
    return observed_epochs, positions, velocities


def _validate_logical_external_identity(value: Any, logical_path: str) -> None:
    if (
        type(value) is not dict
        or set(value) != {"logical_path", "size_bytes", "sha256"}
        or value["logical_path"] != logical_path
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] <= 0
        or not _is_sha256(value["sha256"])
    ):
        raise QualificationError("REBOUND external file identity changed")


def _validate_rebound_runtime_record(value: Any) -> None:
    required = {
        "schema", "classification", "authority_authorized", "distribution_name", "distribution_version", "rebound_version",
        "rebound_githash", "rebound_build", "module_source", "c_library", "python_sources", "distribution_metadata",
        "distribution_license_file", "installed_license_expression", "project_description_mentions_v3_or_later",
        "license_text_inconsistency_observed", "official_repository", "official_license_page",
        "legal_or_redistribution_conclusion_authorized", "rebound_code_vendored", "rebound_code_copied",
        "jx_core_linked_to_rebound", "omp_num_threads",
    }
    if type(value) is not dict or set(value) != required:
        raise QualificationError("REBOUND runtime receipt schema changed")
    if (
        value["schema"] != "jx.wh-v9.qualification.rebound-ias15-runtime.v1"
        or value["classification"]
        != "OPTIONAL_EXTERNAL_GPL_V3_FAMILY_COMPARATOR"
        or value["authority_authorized"] is not False
        or value["distribution_name"] != "rebound"
        or value["distribution_version"] != "5.1.1"
        or value["rebound_version"] != "5.1.1"
        or type(value["rebound_githash"]) is not str
        or not value["rebound_githash"]
        or type(value["rebound_build"]) is not str
        or not value["rebound_build"]
        or value["installed_license_expression"] != "GPL-3.0-only"
        or value["project_description_mentions_v3_or_later"] is not True
        or type(value["license_text_inconsistency_observed"]) is not bool
        or value["official_repository"] != "https://github.com/hannorein/rebound"
        or value["official_license_page"]
        != "https://github.com/hannorein/rebound/blob/5.1.1/LICENSE"
        or any(
            value[key] is not False
            for key in (
                "legal_or_redistribution_conclusion_authorized",
                "rebound_code_vendored",
                "rebound_code_copied",
                "jx_core_linked_to_rebound",
            )
        )
        or value["omp_num_threads"] != "1"
    ):
        raise QualificationError("REBOUND runtime identity/claim boundary changed")
    _validate_logical_external_identity(value["module_source"], "rebound/__init__.py")
    _validate_logical_external_identity(
        value["c_library"], "rebound/NATIVE_LIBREBOUND_BINARY"
    )
    _validate_logical_external_identity(
        value["distribution_metadata"], "rebound-5.1.1.dist-info/METADATA"
    )
    _validate_logical_external_identity(
        value["distribution_license_file"],
        "rebound-5.1.1.dist-info/licenses/LICENSE",
    )
    sources = value["python_sources"]
    if (
        type(sources) is not dict
        or set(sources)
        != {
            "schema", "selection", "classification", "authority_authorized", "required_execution_wrappers", "files",
            "file_count", "tree_hash_algorithm", "tree_hash_domain", "tree_sha256",
        }
        or sources["authority_authorized"] is not False
        or sources["schema"]
        != fixture._PROVENANCE_SUPPORT.REBOUND_SOURCE_MANIFEST_SCHEMA
        or sources["selection"]
        != "ALL_DISTRIBUTION_RECORDED_REBOUND_PACKAGE_PYTHON_SOURCES"
        or sources["classification"] != "PROVENANCE_DIAGNOSTIC"
        or sources["required_execution_wrappers"]
        != [
            "rebound/__init__.py",
            "rebound/integrator.py",
            "rebound/particle.py",
            "rebound/particles.py",
            "rebound/simulation.py",
        ]
        or type(sources["files"]) is not list
        or sources["file_count"] != len(sources["files"])
        or sources["tree_hash_algorithm"]
        != "SHA256_DOMAIN_SEPARATED_SORTED_JSON_V1"
        or sources["tree_hash_domain"]
        != fixture._PROVENANCE_SUPPORT.REBOUND_SOURCE_TREE_HASH_DOMAIN
        or not _is_sha256(sources["tree_sha256"])
    ):
        raise QualificationError("REBOUND Python-source manifest changed")
    paths: list[str] = []
    for record in sources["files"]:
        if (
            type(record) is not dict
            or set(record) != {"path", "size_bytes", "sha256"}
            or type(record["path"]) is not str
            or record["path"].startswith("/")
            or "\\" in record["path"]
            or ".." in Path(record["path"]).parts
            or not record["path"].startswith("rebound/")
            or not record["path"].endswith(".py")
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
            or not _is_sha256(record["sha256"])
        ):
            raise QualificationError("REBOUND Python-source identity changed")
        paths.append(record["path"])
    if (
        paths != sorted(paths)
        or len(paths) != len(set(paths))
        or not set(sources["required_execution_wrappers"]).issubset(paths)
    ):
        raise QualificationError("REBOUND Python-source roster changed")
    expected_tree = fixture._PROVENANCE_SUPPORT._rebound_source_tree_sha256(
        sources["files"]
    )
    if sources["tree_sha256"] != expected_tree:
        raise QualificationError("REBOUND Python-source tree digest changed")


def _expected_particle_readback(
    positions: np.ndarray, velocities: np.ndarray
) -> list[dict[str, float]]:
    return [
        {
            "m": float(fixture.GRAVITATIONAL_PARAMETERS[index]),
            "r": 0.0,
            "x": float(positions[index, 0]),
            "y": float(positions[index, 1]),
            "z": float(positions[index, 2]),
            "vx": float(velocities[index, 0]),
            "vy": float(velocities[index, 1]),
            "vz": float(velocities[index, 2]),
        }
        for index in range(3)
    ]


def _validate_ias15_record(
    record: Any, epochs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(record) is not dict or set(record) != {
        "implementation", "shares_jx_force_or_integrator_code", "runtime", "settings", "accounting", "epochs", "positions", "velocities", "content_sha256"
    }:
        raise QualificationError("IAS15 receipt schema changed")
    core = {key: record[key] for key in record if key != "content_sha256"}
    if (
        record["implementation"]
        != "REBOUND_IAS15_ALL_ACTIVE_NEWTONIAN_POINT_MASSES"
        or record["shares_jx_force_or_integrator_code"] is not False
        or record["content_sha256"]
        != _domain_sha256(IAS15_RECEIPT_DOMAIN, core)
    ):
        raise QualificationError("IAS15 implementation identity changed")
    _validate_rebound_runtime_record(record["runtime"])
    observed_epochs, positions, velocities = _oracle_arrays(record, "IAS15")
    if not _array_bits_equal(observed_epochs, epochs):
        raise QualificationError("IAS15 clock evidence changed")
    initial_positions, initial_velocities = fixture._initial_arrays()
    settings = record["settings"]
    expected_settings = {
        "provenance_categories": {
            "configured_value": ["requested_initial_dt", "requested_initial_dt_hex"],
            "persistent_effective_readback": [
                "integrator", "effective_initial_dt", "effective_initial_dt_hex", "epsilon", "min_dt", "adaptive_mode", "G",
                "gravity", "collision", "boundary", "softening", "N", "N_active", "testparticle_type", "particles_initial_readback",
            ],
            "call_argument": ["integrate(target,exact_finish_time=1)"],
            "environment_observation": ["OMP_NUM_THREADS"],
        },
        "integrator": "ias15",
        "requested_initial_dt": float(fixture.IAS15_INITIAL_DT),
        "requested_initial_dt_hex": fixture.IAS15_INITIAL_DT.hex(),
        "effective_initial_dt": float(fixture.IAS15_INITIAL_DT),
        "effective_initial_dt_hex": fixture.IAS15_INITIAL_DT.hex(),
        "epsilon": 1.0e-13,
        "min_dt": 0.0,
        "adaptive_mode": "PRS23",
        "G": 1.0,
        "gravity": "basic",
        "collision": "none",
        "boundary": "none",
        "softening": 0.0,
        "N": 3,
        "N_active": 3,
        "testparticle_type": 0,
        "particles_initial_readback": _expected_particle_readback(
            initial_positions, initial_velocities
        ),
        "requested_epoch_hex": [float(value).hex() for value in epochs],
        "exact_finish_time": 1,
        "callbacks": "NONE",
        "thread_observation": {"OMP_NUM_THREADS": "1"},
    }
    if settings != expected_settings:
        raise QualificationError("IAS15 settings/input changed")
    accounting = record["accounting"]
    expected_accounting = {
        "checkpoint_count": len(epochs),
        "integrate_target_call_count": len(epochs) - 1,
        "exact_finish_time": 1,
        "final_observed_epoch_hex": float(epochs[-1]).hex(),
        "final_particles_readback": _expected_particle_readback(
            positions[-1], velocities[-1]
        ),
        "evidence_class": "NUMERICAL_REFERENCE_MODEL_OUTPUT",
        "reference_truth_claimed": False,
        "registry_authorized": False,
        "qualification_authorized": False,
    }
    if (
        accounting != expected_accounting
        or not _array_bits_equal(np.ascontiguousarray(positions[0]), initial_positions)
        or not _array_bits_equal(np.ascontiguousarray(velocities[0]), initial_velocities)
    ):
        raise QualificationError("IAS15 accounting/state custody changed")
    return observed_epochs, positions, velocities


def _weak_metric_from_lane(
    lane: dict[str, Any],
    *,
    periods: int,
    expected_divisor: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if type(lane) is not dict or set(lane) != {
        "step_divisor", "candidate", "dop853", "ias15", "metrics_hex", "oracle_difference_hex"
    } or lane["step_divisor"] != expected_divisor:
        raise QualificationError("weak-three-body lane schema/divisor changed")
    profile = fixture.Profile(1, periods, 4)
    snapshot, _ = fixture._state_and_plan(profile)
    spec = fixture._wh_spec(profile, "long", expected_divisor)
    positions, velocities, epochs, result = _bind_v9_receipt_to_request(
        lane["candidate"], snapshot, spec, runtime["v9"]
    )
    dop_epochs, dop_positions, dop_velocities = _validate_dop853_record(
        lane["dop853"], epochs, runtime["scipy"]
    )
    ias = lane["ias15"]
    oracle_difference = None
    if ias is not None:
        ias_epochs, ias_positions, ias_velocities = _validate_ias15_record(
            ias, epochs
        )
        reference_p, reference_v = ias_positions, ias_velocities
        oracle_difference = {
            "position_vector_l2_max": fixture._cartesian_metrics(dop_positions, ias_positions)["vector_l2_max"],
            "velocity_vector_l2_max": fixture._cartesian_metrics(dop_velocities, ias_velocities)["vector_l2_max"],
        }
    else:
        reference_p, reference_v = dop_positions, dop_velocities
    metrics = _weak_lane_metrics(positions, velocities, reference_p, reference_v)
    expected_metrics = {
        "position_vector_l2_max": metrics["position_vector_l2_max"].hex(),
        "velocity_vector_l2_max": metrics["velocity_vector_l2_max"].hex(),
        "phase_error_radians_max": metrics["phase_error_radians_max"].hex(),
        "invariants": {key: float(value).hex() for key, value in metrics["invariants"].items()},
    }
    expected_difference = None if oracle_difference is None else {key: value.hex() for key, value in oracle_difference.items()}
    if lane["metrics_hex"] != expected_metrics or lane["oracle_difference_hex"] != expected_difference:
        raise QualificationError("weak-three-body metrics changed")
    return {"metrics": metrics, "oracle_difference": oracle_difference, "result": result}


def _derive_gates(report: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = _protocol()
    gates: list[dict[str, Any]] = []
    profile = report["profile"]
    analytic_records = report["analytic_two_body"]["cases"]
    analytic_gate = protocol["gates"]["analytic_two_body"]
    maxima = {name: 0.0 for name in (
        "normalized_position_error_max", "normalized_velocity_error_max", "relative_energy_error_max", "relative_angular_momentum_error_max", "normalized_forward_backward_state_error_max"
    )}
    fallback = 0
    for record in analytic_records:
        if type(record) is not dict or set(record) != {
            "case_id", "elements", "step_divisor", "nominal_period_hex", "checkpoint_periods", "candidate", "analytic_oracle", "backward_candidate", "metrics_hex", "fallback_count"
        }:
            raise QualificationError("analytic case schema changed")
        elements = record["elements"]
        if type(elements) is not dict or set(elements) != {
            "semimajor_axis_hex", "eccentricity_hex", "true_anomaly_hex", "secondary_to_primary_gm_ratio_hex", "direction"
        }:
            raise QualificationError("analytic element schema changed")
        case = AnalyticCase(
            float.fromhex(elements["semimajor_axis_hex"]),
            float.fromhex(elements["eccentricity_hex"]),
            float.fromhex(elements["true_anomaly_hex"]),
            float.fromhex(elements["secondary_to_primary_gm_ratio_hex"]),
            elements["direction"],
        )
        expected_checkpoints = tuple(
            protocol["workloads"]["analytic_two_body_portfolio"]["checkpoint_periods"]
            if profile == FORMAL
            else (0, 1)
        )
        expected_period = 2.0 * math.pi * math.sqrt(
            case.semimajor_axis**3 / (1.0 + case.ratio)
        )
        if (
            record["case_id"] != case.case_id
            or record["step_divisor"] != case.step_divisor
            or record["checkpoint_periods"] != list(expected_checkpoints)
            or record["nominal_period_hex"] != expected_period.hex()
        ):
            raise QualificationError("analytic case identity changed")
        snapshot, _, expected_spec, bound_period = _two_body_inputs(
            case, expected_checkpoints
        )
        positions, velocities, epochs, forward_result = _bind_v9_receipt_to_request(
            record["candidate"],
            snapshot,
            expected_spec,
            report["runtime"]["v9"],
        )
        expected_epochs = np.asarray(
            [float((expected_period / case.step_divisor) * (value * case.step_divisor)) for value in expected_checkpoints],
            dtype=np.float64,
        )
        if not _array_bits_equal(epochs, expected_epochs):
            raise QualificationError("analytic candidate epoch lattice changed")
        endpoint_snapshot, _ = _plan_and_snapshot(
            ("PRIMARY", "SECONDARY"),
            positions[-1],
            velocities[-1],
            np.array((1.0, case.ratio), dtype=np.float64),
            epoch=float(epochs[-1]),
            validity_start=-bound_period,
            validity_end=float(epochs[-1] + bound_period),
            fixture_id="jx.wh-v9.qualification.analytic-two-body.reverse.v1",
        )
        reverse_spec = FixedStepWisdomHolmanSpec(
            checkpoint_step_indices=(0, expected_spec.completed_steps),
            fixed_step=-expected_spec.fixed_step,
            maximum_steps=expected_spec.completed_steps,
            jacobi_body_order=("PRIMARY", "SECONDARY"),
            minimum_encounter_pair_separation=expected_spec.minimum_encounter_pair_separation,
            minimum_jacobi_periapse=expected_spec.minimum_jacobi_periapse,
            maximum_initial_barycenter_position_norm=expected_spec.maximum_initial_barycenter_position_norm,
            maximum_initial_barycenter_velocity_norm=expected_spec.maximum_initial_barycenter_velocity_norm,
        )
        back_positions, back_velocities, back_epochs, backward_result = _bind_v9_receipt_to_request(
            record["backward_candidate"],
            endpoint_snapshot,
            reverse_spec,
            report["runtime"]["v9"],
        )
        oracle = record["analytic_oracle"]
        if (
            type(oracle) is not dict
            or set(oracle) != {"implementation", "precision_decimal_digits", "shares_jx_integrator_code", "positions", "velocities"}
            or oracle.get("implementation") != "STANDALONE_DECIMAL_100_DIGIT_ELLIPTIC_KEPLER_SOLUTION"
            or oracle.get("precision_decimal_digits") != 100
            or oracle.get("shares_jx_integrator_code") is not False
        ):
            raise QualificationError("analytic oracle identity changed")
        oracle_p = _decode_array(oracle["positions"], shape=positions.shape)
        oracle_v = _decode_array(oracle["velocities"], shape=velocities.shape)
        expected_oracle_positions: list[np.ndarray] = []
        expected_oracle_velocities: list[np.ndarray] = []
        for epoch in epochs:
            expected_p, expected_v = _analytic_state(case, float(epoch))
            expected_oracle_positions.append(expected_p)
            expected_oracle_velocities.append(expected_v)
        if (
            not _array_bits_equal(oracle_p, np.stack(expected_oracle_positions))
            or not _array_bits_equal(oracle_v, np.stack(expected_oracle_velocities))
            or back_epochs.shape != (2,)
            or float(back_epochs[0]).hex() != float(epochs[-1]).hex()
        ):
            raise QualificationError("analytic oracle or backward clock changed")
        metrics = _two_body_metrics(case, positions, velocities, oracle_p, oracle_v, back_positions, back_velocities)
        if record["metrics_hex"] != {key: value.hex() for key, value in metrics.items()}:
            raise QualificationError("analytic metrics changed")
        for name, value in metrics.items():
            maxima[name] = max(maxima[name], value)
        expected_fallback = forward_result.total_fallback_count + backward_result.total_fallback_count
        if type(record["fallback_count"]) is not int or record["fallback_count"] != expected_fallback:
            raise QualificationError("analytic fallback accounting changed")
        fallback += record["fallback_count"]
    _gate_bool(gates, "analytic.portfolio.case_count", len(analytic_records) == 240)
    for name, value in maxima.items():
        _gate_le(gates, f"analytic.{name}", value, float(analytic_gate[name]))
    _gate_bool(gates, "analytic.fallback_count_zero", fallback == analytic_gate["fallback_count_required"])

    weak = report["weak_three_body"]
    assessed: dict[str, list[dict[str, Any]]] = {}
    for horizon in ("short", "long"):
        container = weak[horizon]
        if type(container) is not dict or set(container) != {"periods", "lanes"}:
            raise QualificationError("weak-three-body horizon container changed")
        expected_periods = (
            (1 if horizon == "short" else 2)
            if profile == SMOKE
            else protocol["workloads"]["weak_three_body"][
                f"{horizon}_horizon_periods"
            ]
        )
        if (
            type(container["periods"]) is not int
            or container["periods"] != expected_periods
        ):
            raise QualificationError("weak-three-body horizon periods changed")
        lanes = container["lanes"]
        if (
            type(lanes) is not list
            or len(lanes) != 3
            or any(type(lane) is not dict for lane in lanes)
            or [lane.get("step_divisor") for lane in lanes] != [64, 128, 256]
        ):
            raise QualificationError("weak-three-body lane roster changed")
        assessed[horizon] = [
            _weak_metric_from_lane(
                lane,
                periods=container["periods"],
                expected_divisor=divisor,
                runtime=report["runtime"],
            )
            for divisor, lane in zip((64, 128, 256), lanes, strict=True)
        ]
        convergence = protocol["gates"]["weak_three_body_convergence"]
        for metric, prefix in (
            ("position_vector_l2_max", "position"),
            ("velocity_vector_l2_max", "velocity"),
            ("phase_error_radians_max", "phase"),
        ):
            values = [item["metrics"][metric] for item in assessed[horizon]]
            for left, right, label in ((values[0], values[1], "p64_over_p128"), (values[1], values[2], "p128_over_p256")):
                ratio = left / right if right > 0.0 else math.inf
                _gate_range(gates, f"weak.{horizon}.convergence.{prefix}.{label}", ratio, float(convergence[f"adjacent_{prefix}_error_ratio_min"]), float(convergence[f"adjacent_{prefix}_error_ratio_max"]))
        oracle_gate = protocol["gates"]["weak_three_body_oracle_floor"]
        p256_metrics = assessed[horizon][2]["metrics"]
        for divisor, item in zip((64, 128, 256), assessed[horizon], strict=True):
            difference = item["oracle_difference"]
            _gate_bool(gates, f"weak.{horizon}.p{divisor}.ias15_present", difference is not None)
            if difference is not None:
                for metric, candidate_metric in (("position_vector_l2_max", "position_vector_l2_max"), ("velocity_vector_l2_max", "velocity_vector_l2_max")):
                    value = difference[metric]
                    _gate_le(gates, f"weak.{horizon}.p{divisor}.oracle_floor.{metric}", value, float(oracle_gate["position_and_velocity_oracle_difference_max"]))
                    p256_error = p256_metrics[candidate_metric]
                    fraction = value / p256_error if p256_error > 0.0 else math.inf
                    _gate_le(gates, f"weak.{horizon}.p{divisor}.oracle_fraction.{metric}", fraction, float(oracle_gate["oracle_difference_fraction_of_v9_p256_error_max"]))

    absolute = protocol["gates"]["weak_three_body_absolute"]
    for divisor, scale, item in zip((64, 128, 256), (1.0, float(absolute["p128_scale_from_p64"]), float(absolute["p256_scale_from_p64"])), assessed["long"], strict=True):
        metrics = item["metrics"]
        _gate_le(gates, f"weak.long.p{divisor}.position", metrics["position_vector_l2_max"], float(absolute["p64_position_error_max"]) * scale)
        _gate_le(gates, f"weak.long.p{divisor}.velocity", metrics["velocity_vector_l2_max"], float(absolute["p64_velocity_error_max"]) * scale)
        _gate_le(gates, f"weak.long.p{divisor}.phase", metrics["phase_error_radians_max"], float(absolute["p64_phase_error_radians_max"]) * scale)
        invariant = metrics["invariants"]
        _gate_le(gates, f"weak.long.p{divisor}.energy", invariant["relative_total_energy_max"], float(absolute["p64_relative_energy_error_max"]) * scale)
        invariant_thresholds = {
            "relative_angular_momentum_max": "relative_angular_momentum_error_max",
            "center_of_mass_position_max": "center_of_mass_position_max",
            "total_momentum_max": "total_momentum_max",
        }
        for name, threshold_name in invariant_thresholds.items():
            _gate_le(gates, f"weak.long.p{divisor}.{name}", invariant[name], float(absolute[threshold_name]))
        _gate_bool(gates, f"weak.long.p{divisor}.fallback_zero", item["result"].total_fallback_count == 0)
    roundtrip = weak["p128_roundtrip"]
    if type(roundtrip) is not dict or set(roundtrip) != {
        "forward", "backward", "normalized_state_error_hex"
    }:
        raise QualificationError("weak-three-body roundtrip schema changed")
    roundtrip_periods = weak["long"]["periods"]
    roundtrip_profile = fixture.Profile(1, roundtrip_periods, 4)
    initial_snapshot, _ = fixture._state_and_plan(roundtrip_profile)
    forward_spec = fixture._wh_spec(roundtrip_profile, "long", 128)
    forward_positions, forward_velocities, forward_epochs, forward = (
        _bind_v9_receipt_to_request(
            roundtrip["forward"],
            initial_snapshot,
            forward_spec,
            report["runtime"]["v9"],
        )
    )
    endpoint_snapshot, _ = _plan_and_snapshot(
        fixture.BODY_IDS,
        forward_positions[-1],
        forward_velocities[-1],
        fixture.GRAVITATIONAL_PARAMETERS,
        epoch=float(forward_epochs[-1]),
        validity_start=-fixture.PERIOD,
        validity_end=float(forward_epochs[-1] + fixture.PERIOD),
        fixture_id="jx.wh-v9.qualification.weak-three-body.reverse.v1",
    )
    reverse_spec = FixedStepWisdomHolmanSpec(
        checkpoint_step_indices=(0, forward_spec.completed_steps),
        fixed_step=-forward_spec.fixed_step,
        maximum_steps=forward_spec.completed_steps,
        jacobi_body_order=fixture.BODY_IDS,
        minimum_encounter_pair_separation=0.01,
        minimum_jacobi_periapse=0.05,
        maximum_initial_barycenter_position_norm=1.0e-12,
        maximum_initial_barycenter_velocity_norm=1.0e-12,
    )
    backward_positions, backward_velocities, _, backward = (
        _bind_v9_receipt_to_request(
            roundtrip["backward"],
            endpoint_snapshot,
            reverse_spec,
            report["runtime"]["v9"],
        )
    )
    position_scale = float(
        np.max(np.linalg.norm(initial_snapshot.positions, axis=1))
    )
    velocity_scale = float(
        np.max(np.linalg.norm(initial_snapshot.velocities, axis=1))
    )
    recomputed_roundtrip = max(
        float(
            np.max(
                np.linalg.norm(
                    backward_positions[-1] - initial_snapshot.positions, axis=1
                )
            )
            / position_scale
        ),
        float(
            np.max(
                np.linalg.norm(
                    backward_velocities[-1] - initial_snapshot.velocities, axis=1
                )
            )
            / velocity_scale
        ),
    )
    if (
        type(roundtrip["normalized_state_error_hex"]) is not str
        or roundtrip["normalized_state_error_hex"] != recomputed_roundtrip.hex()
    ):
        raise QualificationError("weak-three-body roundtrip scalar changed")
    _gate_le(gates, "weak.long.p128.roundtrip", recomputed_roundtrip, float(absolute["p128_normalized_forward_backward_state_error_max"]))
    _gate_bool(gates, "weak.long.p128.roundtrip_fallback_zero", forward.total_fallback_count + backward.total_fallback_count == 0)

    invariant_gate = protocol["gates"]["long_sampled_invariants"]
    invariant_values: list[dict[str, float]] = []
    invariant_container = weak["invariant_horizon"]
    if type(invariant_container) is not dict or set(invariant_container) != {
        "periods", "samples_per_period", "finite_horizon_observation_only", "lanes"
    }:
        raise QualificationError("long-invariant container schema changed")
    if (
        type(invariant_container["periods"]) is not int
        or invariant_container["periods"]
        != (
            2
            if profile == SMOKE
            else protocol["workloads"]["weak_three_body"][
                "invariant_horizon_periods"
            ]
        )
        or invariant_container["samples_per_period"] != 4
        or invariant_container["finite_horizon_observation_only"] is not True
        or type(invariant_container["lanes"]) is not list
        or [
            lane.get("step_divisor") if type(lane) is dict else None
            for lane in invariant_container["lanes"]
        ]
        != [64, 128]
    ):
        raise QualificationError("long-invariant roster/settings changed")
    invariant_profile = fixture.Profile(1, invariant_container["periods"], 4)
    invariant_snapshot, _ = fixture._state_and_plan(invariant_profile)
    for divisor, lane in zip(
        (64, 128), invariant_container["lanes"], strict=True
    ):
        if type(lane) is not dict or set(lane) != {
            "step_divisor", "candidate", "metrics_hex"
        }:
            raise QualificationError("long-invariant lane schema changed")
        invariant_spec = fixture._wh_spec(invariant_profile, "long", divisor)
        positions, velocities, _, result = _bind_v9_receipt_to_request(
            lane["candidate"],
            invariant_snapshot,
            invariant_spec,
            report["runtime"]["v9"],
        )
        metrics = fixture._invariant_metrics(positions, velocities)
        if lane["metrics_hex"] != {key: float(value).hex() for key, value in metrics.items()}:
            raise QualificationError("long invariant metrics changed")
        invariant_values.append(metrics)
        energy_threshold = float(invariant_gate[f"p{divisor}_relative_energy_envelope_max"])
        _gate_le(gates, f"weak.invariant.p{divisor}.energy", metrics["relative_total_energy_max"], energy_threshold)
        invariant_thresholds = {
            "relative_angular_momentum_max": "relative_angular_momentum_error_max",
            "center_of_mass_position_max": "center_of_mass_position_max",
            "total_momentum_max": "total_momentum_max",
        }
        for name, threshold_name in invariant_thresholds.items():
            _gate_le(gates, f"weak.invariant.p{divisor}.{name}", metrics[name], float(invariant_gate[threshold_name]))
        _gate_bool(gates, f"weak.invariant.p{divisor}.fallback_zero", result.total_fallback_count == 0)
    ratio = (
        invariant_values[0]["relative_total_energy_max"]
        / invariant_values[1]["relative_total_energy_max"]
        if invariant_values[1]["relative_total_energy_max"] > 0.0
        else math.inf
    )
    _gate_range(gates, "weak.invariant.energy_ratio_p64_over_p128", ratio, float(invariant_gate["p64_over_p128_energy_envelope_ratio_min"]), float(invariant_gate["p64_over_p128_energy_envelope_ratio_max"]))

    formal_spec = protocol["workloads"]["weak_three_body"]
    _gate_bool(gates, "custody.formal_profile", profile == FORMAL)
    _gate_bool(gates, "custody.short_horizon", weak["short"]["periods"] == formal_spec["short_horizon_periods"])
    _gate_bool(gates, "custody.long_horizon", weak["long"]["periods"] == formal_spec["long_horizon_periods"])
    _gate_bool(gates, "custody.invariant_horizon", weak["invariant_horizon"]["periods"] == formal_spec["invariant_horizon_periods"])
    _gate_bool(gates, "custody.scipy_exact", report["runtime"]["scipy"]["version"] == protocol["independent_oracles"]["dop853"]["required_scipy_version"])
    _gate_bool(gates, "custody.clean_git_head_tree", report["sources"]["git_tree_verified"] is True)
    try:
        validated_engineering = _engineering_receipt(
                report["engineering_evidence"],
                protocol,
                report["sources"],
                report["runtime"],
            )
        engineering_ok = (
            validated_engineering is not None
            and validated_engineering["formal_execution_verification"]
            is not None
            and validated_engineering["normal_python"][
                "python_executable_sha256"
            ]
            == report["runtime"]["python"]["executable_sha256"]
            and validated_engineering["optimized_python"][
                "python_executable_sha256"
            ]
            == report["runtime"]["python"]["executable_sha256"]
        )
    except QualificationError:
        engineering_ok = False
    _gate_bool(gates, "custody.engineering_prerequisites", engineering_ok)
    return gates


def _run_invariant_horizon(periods: int) -> dict[str, Any]:
    lanes: list[dict[str, Any]] = []
    for divisor in (64, 128):
        candidate = _run_weak_v9(periods, divisor)
        positions, velocities, _, _ = _validate_v9_receipt(candidate)
        metrics = fixture._invariant_metrics(positions, velocities)
        lanes.append(
            {
                "step_divisor": divisor,
                "candidate": candidate,
                "metrics_hex": {key: float(value).hex() for key, value in metrics.items()},
            }
        )
    return {"periods": periods, "samples_per_period": 4, "finite_horizon_observation_only": True, "lanes": lanes}


def run_qualification(
    profile: str = SMOKE,
    *,
    engineering_evidence: dict[str, Any] | None = None,
    engineering_paths: EngineeringEvidencePaths | None = None,
    include_ias15: bool | None = None,
) -> dict[str, Any]:
    """Execute a smoke diagnostic or the exact registered formal matrix.

    ``formal`` is intentionally expensive and requires exact SciPy/REBOUND
    dependencies plus a bound engineering receipt.  ``smoke`` exercises the
    same implementations on a reduced matrix and can never become eligible.
    """

    if type(profile) is not str or profile not in PROFILE_NAMES:
        raise QualificationError(f"profile must be one of {PROFILE_NAMES!r}")
    if engineering_evidence is not None and engineering_paths is not None:
        raise QualificationError(
            "engineering receipt and path-backed evidence are mutually exclusive"
        )
    if profile == SMOKE and (
        engineering_evidence is not None or engineering_paths is not None
    ):
        raise QualificationError(
            "smoke execution does not accept formal engineering evidence"
        )
    if profile == FORMAL and engineering_evidence is not None:
        raise QualificationError(
            "formal execution rejects standalone engineering receipts; "
            "path-backed logs and build artifacts are required"
        )
    if profile == FORMAL and engineering_paths is None:
        raise QualificationError(
            "formal execution requires path-backed normal/-OO logs and "
            "two independently built wheels and sdists"
        )
    if include_ias15 is None:
        include_ias15 = profile == FORMAL
    if type(include_ias15) is not bool:
        raise QualificationError("include_ias15 must be a built-in bool")
    if profile == FORMAL and include_ias15 is not True:
        raise QualificationError("formal execution requires REBOUND IAS15")
    protocol = _protocol()
    source_custody = _source_custody(require_clean=profile == FORMAL)
    if profile == FORMAL:
        _validate_formal_loaded_source_modules(source_custody)
    runtime_custody = _runtime_custody()
    engineering = (
        _derive_engineering_receipt(
            engineering_paths,
            protocol,
            source_custody,
            runtime_custody,
            formal_verify=profile == FORMAL,
        )
        if engineering_paths is not None
        else _engineering_receipt(
            engineering_evidence,
            protocol,
            source_custody,
            runtime_custody,
        )
    )
    if profile == FORMAL:
        _require_unchanged_live_custody(
            source_custody, runtime_custody, "during engineering verification"
        )
    analytic_checkpoints = tuple(protocol["workloads"]["analytic_two_body_portfolio"]["checkpoint_periods"])
    if profile == SMOKE:
        analytic_checkpoints = (0, 1)
        short_periods, long_periods, invariant_periods = 1, 2, 2
    else:
        weak_spec = protocol["workloads"]["weak_three_body"]
        short_periods = weak_spec["short_horizon_periods"]
        long_periods = weak_spec["long_horizon_periods"]
        invariant_periods = weak_spec["invariant_horizon_periods"]

    science_context = nullcontext()
    if profile == FORMAL:
        if engineering_paths is None or engineering is None:
            raise QualificationError("formal wheel-science evidence is absent")
        science_context = _formal_science_native_from_wheel(
            engineering_paths.wheel_build_a,
            engineering["formal_execution_verification"][
                "science_runtime_loading"
            ],
            source_custody,
            runtime_custody,
        )
    with science_context:
        if profile == FORMAL:
            _require_unchanged_live_custody(
                source_custody,
                runtime_custody,
                "before wheel-loaded scientific execution",
            )
        cases = [
            _run_analytic_case(case, analytic_checkpoints)
            for case in _analytic_cases(profile, protocol)
        ]
        weak = {
            "fixture_id": protocol["workloads"]["weak_three_body"]["fixture_id"],
            "initial_state_sha256": fixture._initial_state_sha256(),
            "short": _run_weak_horizon(short_periods, (64, 128, 256), require_exact_scipy=profile == FORMAL, include_ias15=include_ias15),
            "long": _run_weak_horizon(long_periods, (64, 128, 256), require_exact_scipy=profile == FORMAL, include_ias15=include_ias15),
            "p128_roundtrip": _run_weak_roundtrip(long_periods, 128),
            "invariant_horizon": _run_invariant_horizon(invariant_periods),
        }
        if profile == FORMAL:
            _require_unchanged_live_custody(
                source_custody,
                runtime_custody,
                "during wheel-loaded scientific execution",
            )
    if weak["initial_state_sha256"] != protocol["workloads"]["weak_three_body"]["initial_state_sha256"]:
        raise QualificationError("locked weak-three-body fixture changed")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "qualification_id": protocol["qualification_id"],
        "protocol": {
            "sha256": PROTOCOL_SHA256,
            "revision": protocol["protocol_revision"],
            "registration_commit": PROTOCOL_REGISTRATION_COMMIT,
            "base_repository_commit": protocol["base_repository_commit"],
        },
        "profile": profile,
        "sources": source_custody,
        "runtime": runtime_custody,
        "analytic_two_body": {
            "case_count": len(cases),
            "checkpoint_periods": list(analytic_checkpoints),
            "cases": cases,
        },
        "weak_three_body": weak,
        "engineering_evidence": engineering,
        "gates": [],
        "decision": {},
        "claim_ceiling": protocol["claim_ceiling"],
        "claim_controls": protocol["claim_controls"],
        "scientific_interpretation": {
            "intrinsic_structural_integrity_verified": True,
            "repository_custody_requires_live_reproduction_check": True,
            "internal_decision_is_final_approval": False,
            "external_approval_required": True,
            "second_independently_administered_host_required": True,
            "named_external_reviewer_signature_required": True,
            "v9_remains_screening_only": True,
        },
        "content_integrity": {},
    }
    report["gates"] = _derive_gates(report)
    all_passed = all(gate["passed"] is True for gate in report["gates"])
    report["decision"] = {
        "status": INTERNAL_PASS if all_passed else INTERNAL_FAIL,
        "all_numeric_and_custody_gates_conjunctive": True,
        "all_gates_passed": all_passed,
        "scientific_approval_granted": False,
        "external_review_completed": False,
        "next_gate": protocol["next_gate"],
    }
    report["content_integrity"] = {
        "algorithm": "SHA256_DOMAIN_SEPARATED_CANONICAL_ASCII_JSON_V1",
        "domain": REPORT_DOMAIN,
        "sha256": _domain_sha256(REPORT_DOMAIN, {key: value for key, value in report.items() if key != "content_integrity"}),
    }
    validate_report(report)
    if profile == FORMAL:
        _require_unchanged_live_custody(
            source_custody, runtime_custody, "during scientific execution"
        )
    return report


def _validate_json_tree(value: Any) -> None:
    nodes = 0

    def walk(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise QualificationError("JSON tree exceeds its structural cap")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise QualificationError("JSON object key is not a string")
                walk(child, depth + 1)
        elif type(item) is list:
            for child in item:
                walk(child, depth + 1)
        elif type(item) is float:
            if not math.isfinite(item):
                raise QualificationError("JSON tree contains a nonfinite float")
        elif type(item) not in (str, int, bool, type(None)):
            raise QualificationError("JSON tree contains an unsupported type")

    walk(value, 0)


def validate_report(report: Any) -> None:
    _validate_json_tree(report)
    required = {
        "schema", "qualification_id", "protocol", "profile", "sources", "runtime", "analytic_two_body", "weak_three_body", "engineering_evidence", "gates", "decision", "claim_ceiling", "claim_controls", "scientific_interpretation", "content_integrity"
    }
    if type(report) is not dict or set(report) != required:
        raise QualificationError("qualification-report schema changed")
    protocol = _protocol()
    if report["schema"] != SCHEMA or report["qualification_id"] != protocol["qualification_id"]:
        raise QualificationError("qualification-report identity changed")
    if report["protocol"] != {
        "sha256": PROTOCOL_SHA256,
        "revision": 3,
        "registration_commit": PROTOCOL_REGISTRATION_COMMIT,
        "base_repository_commit": protocol["base_repository_commit"],
    }:
        raise QualificationError("qualification protocol binding changed")
    if report["profile"] not in PROFILE_NAMES:
        raise QualificationError("qualification profile changed")
    _validate_intrinsic_sources(report["sources"], protocol, report["profile"])
    _validate_intrinsic_runtime(report["runtime"], protocol)
    _validate_source_runtime_closure(report["sources"], report["runtime"])
    expected_cases = _analytic_cases(report["profile"], protocol)
    expected_checkpoints = (
        protocol["workloads"]["analytic_two_body_portfolio"][
            "checkpoint_periods"
        ]
        if report["profile"] == FORMAL
        else [0, 1]
    )
    analytic = report["analytic_two_body"]
    if (
        type(analytic) is not dict
        or set(analytic) != {"case_count", "checkpoint_periods", "cases"}
        or analytic["case_count"] != len(expected_cases)
        or analytic["checkpoint_periods"] != expected_checkpoints
        or type(analytic["cases"]) is not list
        or len(analytic["cases"]) != len(expected_cases)
    ):
        raise QualificationError("analytic portfolio schema or cardinality changed")
    for expected, retained in zip(expected_cases, analytic["cases"], strict=True):
        if retained.get("case_id") != expected.case_id:
            raise QualificationError("analytic portfolio order changed")
    weak = report["weak_three_body"]
    if type(weak) is not dict or set(weak) != {"fixture_id", "initial_state_sha256", "short", "long", "p128_roundtrip", "invariant_horizon"}:
        raise QualificationError("weak-three-body schema changed")
    if weak["fixture_id"] != protocol["workloads"]["weak_three_body"]["fixture_id"] or weak["initial_state_sha256"] != fixture.INITIAL_STATE_SHA256:
        raise QualificationError("weak-three-body fixture changed")
    expected_gates = _derive_gates(report)
    if report["gates"] != expected_gates:
        raise QualificationError("qualification gates changed")
    all_passed = all(gate["passed"] is True for gate in expected_gates)
    expected_decision = {
        "status": INTERNAL_PASS if all_passed else INTERNAL_FAIL,
        "all_numeric_and_custody_gates_conjunctive": True,
        "all_gates_passed": all_passed,
        "scientific_approval_granted": False,
        "external_review_completed": False,
        "next_gate": protocol["next_gate"],
    }
    if report["decision"] != expected_decision or report["decision"]["status"] not in (INTERNAL_PASS, INTERNAL_FAIL):
        raise QualificationError("qualification decision changed")
    if report["claim_ceiling"] != protocol["claim_ceiling"] or report["claim_controls"] != protocol["claim_controls"]:
        raise QualificationError("qualification claim ceiling changed")
    if report["scientific_interpretation"] != {
        "intrinsic_structural_integrity_verified": True,
        "repository_custody_requires_live_reproduction_check": True,
        "internal_decision_is_final_approval": False,
        "external_approval_required": True,
        "second_independently_administered_host_required": True,
        "named_external_reviewer_signature_required": True,
        "v9_remains_screening_only": True,
    }:
        raise QualificationError("scientific interpretation changed")
    integrity = report["content_integrity"]
    expected_hash = _domain_sha256(REPORT_DOMAIN, {key: value for key, value in report.items() if key != "content_integrity"})
    if integrity != {"algorithm": "SHA256_DOMAIN_SEPARATED_CANONICAL_ASCII_JSON_V1", "domain": REPORT_DOMAIN, "sha256": expected_hash}:
        raise QualificationError("qualification report content digest changed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError("duplicate JSON object key")
        result[key] = value
    return result


def load_report_bytes(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes or not data or len(data) > MAX_REPORT_BYTES:
        raise QualificationError("report bytes are empty or exceed the cap")
    try:
        text = data.decode("ascii")
        value = json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(QualificationError(f"nonfinite JSON token {token}")),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise QualificationError("report is not strict ASCII JSON") from exc
    if type(value) is not dict:
        raise QualificationError("report root is not an object")
    canonical = _canonical_bytes(value)
    if data not in (canonical, canonical + b"\n"):
        raise QualificationError("report bytes are not canonical sorted JSON")
    validate_report(value)
    return value


def report_bytes(report: dict[str, Any]) -> bytes:
    validate_report(report)
    data = _canonical_bytes(report, newline=True)
    if len(data) > MAX_REPORT_BYTES:
        raise QualificationError("qualification report exceeds the byte cap")
    return data


def _open_nofollow_parent(absolute: str) -> tuple[int, str]:
    parent_path, leaf = os.path.split(absolute)
    if leaf in ("", ".", ".."):
        raise OSError("output leaf is invalid")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd = os.open(os.path.sep, directory_flags)
    try:
        for component in Path(parent_path).parts[1:]:
            next_fd = os.open(
                component, directory_flags, dir_fd=directory_fd
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd, leaf
    except BaseException:
        os.close(directory_fd)
        raise


def _verify_published_leaf(
    absolute: str,
    expected_parent_identity: tuple[int, int],
    expected_leaf_identity: tuple[int, int],
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Re-walk without following links and finish on a leaf identity check."""

    directory_fd, leaf = _open_nofollow_parent(absolute)
    leaf_fd: int | None = None
    try:
        parent = os.fstat(directory_fd)
        if (parent.st_dev, parent.st_ino) != expected_parent_identity:
            raise OSError("output ancestry changed during publication")
        leaf_fd = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(leaf_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != expected_leaf_identity
            or opened.st_size != expected_size
        ):
            raise OSError("output leaf identity changed during publication")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(leaf_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise OSError("output leaf content changed during publication")
        # This lstat-equivalent is deliberately the final verification after
        # the parent-directory fsync.  It ensures the pathname still names the
        # exact inode whose bytes were just hashed.
        retained = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if (retained.st_dev, retained.st_ino) != expected_leaf_identity:
            raise OSError("output leaf changed after durable publication")
    finally:
        if leaf_fd is not None:
            os.close(leaf_fd)
        os.close(directory_fd)


def _write_new_file(path: Path, data: bytes) -> None:
    """Durably create a leaf below a pinned, no-follow directory chain."""

    if not isinstance(path, Path) or type(data) is not bytes:
        raise QualificationError("output path/data types changed")
    raw_path = os.fspath(path)
    if not raw_path or "\0" in raw_path:
        raise QualificationError("output path is invalid")
    absolute = os.path.abspath(raw_path)
    directory_fd: int | None = None
    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    leaf = ""
    try:
        directory_fd, leaf = _open_nofollow_parent(absolute)
        parent = os.fstat(directory_fd)
        parent_identity = (parent.st_dev, parent.st_ino)
        try:
            os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise QualificationError("output path must not exist")
        create_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(
            leaf, create_flags, 0o600, dir_fd=directory_fd
        )
        created = os.fstat(descriptor)
        created_identity = (created.st_dev, created.st_ino)
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short create-only output write")
            offset += written
        os.fsync(descriptor)
        # Keep this descriptor open until publication verification completes.
        # Besides pinning the bytes that were fsynced, the open reference
        # prevents an unlinked created inode from being recycled before the
        # failure cleanup decides whether the pathname still names it.
        os.fsync(directory_fd)
        _verify_published_leaf(
            absolute,
            parent_identity,
            created_identity,
            len(data),
            _sha256(data),
        )
    except QualificationError:
        raise
    except (FileExistsError, NotADirectoryError) as exc:
        raise QualificationError(
            "output path must not exist and ancestry must be real directories"
        ) from exc
    except OSError as exc:
        if directory_fd is not None and created_identity is not None:
            try:
                retained = os.stat(
                    leaf, dir_fd=directory_fd, follow_symlinks=False
                )
                if (retained.st_dev, retained.st_ino) == created_identity:
                    os.unlink(leaf, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except OSError:
                pass
        raise QualificationError("qualification report could not be durably created") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)


def _load_json_file(
    path: Path, *, maximum_bytes: int = MAX_ENGINEERING_JSON_BYTES
) -> dict[str, Any]:
    data, _ = _read_bounded_regular_file(path, maximum_bytes, "input JSON")
    try:
        value = json.loads(data.decode("ascii"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise QualificationError("input is not strict ASCII JSON") from exc
    if type(value) is not dict or data not in (_canonical_bytes(value), _canonical_bytes(value, newline=True)):
        raise QualificationError("input JSON is not canonical")
    _validate_json_tree(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, default=SMOKE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--create-engineering-receipt",
        type=Path,
        help="run fixed normal/-OO tests and create an ancillary receipt only",
    )
    parser.add_argument("--normal-log", type=Path)
    parser.add_argument("--optimized-log", type=Path)
    parser.add_argument("--wheel-build-a", type=Path)
    parser.add_argument("--wheel-build-b", type=Path)
    parser.add_argument("--sdist-build-a", type=Path)
    parser.add_argument("--sdist-build-b", type=Path)
    parser.add_argument("--acknowledge-formal-cost", action="store_true")
    parser.add_argument("--smoke-with-ias15", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--verify-live", action="store_true")
    arguments = parser.parse_args(argv)
    evidence_values = (
        arguments.normal_log,
        arguments.optimized_log,
        arguments.wheel_build_a,
        arguments.wheel_build_b,
        arguments.sdist_build_a,
        arguments.sdist_build_b,
    )
    if any(value is not None for value in evidence_values) and not all(
        value is not None for value in evidence_values
    ):
        raise QualificationError(
            "all six engineering log/artifact paths are required together"
        )
    engineering_paths = (
        EngineeringEvidencePaths(*evidence_values)  # type: ignore[arg-type]
        if all(value is not None for value in evidence_values)
        else None
    )
    if arguments.verify is not None:
        if (
            arguments.output is not None
            or arguments.create_engineering_receipt is not None
        ):
            raise QualificationError("verification cannot be combined with execution")
        retained_bytes, _ = _read_bounded_regular_file(
            arguments.verify, MAX_REPORT_BYTES, "retained qualification report"
        )
        retained = load_report_bytes(retained_bytes)
        if arguments.verify_live:
            if (
                retained["profile"] == FORMAL
                and not arguments.acknowledge_formal_cost
            ):
                raise QualificationError(
                    "formal live reproduction requires --acknowledge-formal-cost"
                )
            validate_live_reproduction(
                retained, engineering_paths=engineering_paths
            )
        elif engineering_paths is not None:
            raise QualificationError(
                "engineering paths are only used by --verify-live"
            )
        return 0
    if arguments.verify_live:
        raise QualificationError("--verify-live requires --verify")
    if arguments.create_engineering_receipt is not None:
        if arguments.output is not None:
            raise QualificationError(
                "engineering receipt creation never runs a scientific report"
            )
        if engineering_paths is None:
            raise QualificationError(
                "engineering receipt creation requires log and artifact paths"
            )
        create_engineering_receipt(
            engineering_paths, arguments.create_engineering_receipt
        )
        return 0
    if arguments.output is None:
        raise QualificationError("execution requires --output")
    if arguments.profile == FORMAL and not arguments.acknowledge_formal_cost:
        raise QualificationError("formal profile requires --acknowledge-formal-cost")
    report = run_qualification(
        arguments.profile,
        engineering_paths=engineering_paths,
        include_ias15=True if arguments.profile == FORMAL else arguments.smoke_with_ias15,
    )
    _write_new_file(arguments.output, report_bytes(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationError as exc:
        print(f"qualification error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
