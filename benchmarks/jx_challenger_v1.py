#!/usr/bin/env python3
"""Run the smallest auditable JX-versus-REBOUND comparison bundle.

Challenger V1 is an orchestrator, not a fourth integrator.  It executes three
existing, self-validating benchmark programs in separate isolated CPython
processes, validates their finite JSON reports, and publishes one compact
aggregate beside the unmodified child reports.  V1 intentionally exposes only
the locked smoke workloads.  It never scores or ranks methods.

The default disables REBOUND, so the runner remains useful in a JX-only source
checkout.  ``auto`` and ``required`` accept only the exact REBOUND 5.1.1
identity already enforced by each child benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "jx.challenger.smoke-bundle.v1"
BENCHMARK_ID = "jx.challenger.solver-portfolio.smoke.v1"
PROFILE_NAME = "smoke"
REBOUND_MODES = ("disabled", "auto", "required")
REQUIRED_REBOUND_VERSION = "5.1.1"
EXPECTED_ENGINE_SOURCE_TREE_SHA256 = (
    "974e31567db7f0e717f76944aedb0c027632f9ab2f4f9f4cbf9b66866d114269"
)

REPORT_FILENAME = "challenger_v1.json"
CHECKSUM_FILENAME = "SHA256SUMS"
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_CHILD_STDOUT_BYTES = 4 * 1024 * 1024
MAX_CHILD_STDERR_BYTES = 256 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 250_000
MAX_JSON_STRING_BYTES = 1024 * 1024
MAX_JSON_INTEGER_BITS = 4096
CHILD_TIMEOUT_SECONDS = 300.0

_SEMANTIC_DOMAIN = "jx.challenger.semantic-content.v1"
_CONTENT_DOMAIN = "jx.challenger.aggregate-content.v1"
_STUDY_DOMAIN = "jx.challenger.study-summary.v1"
_LANE_DOMAIN = "jx.challenger.lane-summary.v1"

_AGGREGATE_CLAIM_CONTROLS = {
    "authority_authorized": False,
    "cross_study_ranking_authorized": False,
    "production_use_authorized": False,
    "qualification_claimed": False,
    "reference_truth_claimed": False,
    "registry_authorized": False,
    "superiority_claimed": False,
    "timing_comparable": False,
}

_BOOTSTRAP = (
    "import runpy,sys;"
    "sys.path.insert(0,sys.argv[1]);"
    "script=sys.argv[2];"
    "sys.argv=[script,*sys.argv[3:]];"
    "runpy.run_path(script,run_name='__main__')"
)


class ChallengerError(RuntimeError):
    """Raised when Challenger cannot construct a complete validated bundle."""


@dataclass(frozen=True, slots=True)
class StudySpec:
    study_id: str
    script_relative_path: str
    child_arguments: tuple[str, ...]
    report_filename: str
    child_schema: str
    child_benchmark_id: str
    child_profile_id: str | None
    script_size_bytes: int
    script_sha256: str
    exact_top_keys: frozenset[str]


_STUDIES = (
    StudySpec(
        "leapfrog_binary",
        "benchmarks/rebound_leapfrog_comparison.py",
        ("--periods", "1", "--samples-per-period", "16"),
        "leapfrog_smoke.json",
        "jx.rebound_leapfrog.equal_mass_binary.v1",
        "jx.rebound_leapfrog.newtonian_equal_mass_binary.v1",
        "jx.rebound_leapfrog.equal_mass_binary.smoke.v1",
        76_043,
        "dca1f9862e4b077996398536b481ab8fc38c043159876fbe53fbec74550f34db",
        frozenset(
            {
                "analytic_workload_gates",
                "benchmark_id",
                "checkpoints",
                "claim_controls",
                "cross_lane_disagreement",
                "lanes",
                "problem",
                "profile",
                "runtime",
                "schema",
                "scientific_interpretation",
                "serialization",
            }
        ),
    ),
    StudySpec(
        "whfast_weak_three_body",
        "benchmarks/rebound_whfast_comparison.py",
        ("--profile", "smoke"),
        "whfast_smoke.json",
        "jx.rebound_whfast.weak_three_body.v1",
        "jx.rebound_whfast.ordered_jacobi_weak_three_body.v1",
        "jx.rebound_whfast.ordered_jacobi_weak_three_body.v1.smoke.v1",
        72_936,
        "2a3b233eeda1ec3b99e4bee6e999ead6753c9d51e996f2282919a0057ff4123d",
        frozenset(
            {
                "benchmark_id",
                "claim_controls",
                "ias15_numerical_references",
                "interpretation",
                "lanes",
                "long_descriptive_envelope",
                "method_provenance",
                "negative_domain_control",
                "problem",
                "profile",
                "runtime_provenance",
                "schema",
                "serialization",
                "short_convergence",
                "timing_disclosure",
            }
        ),
    ),
    StudySpec(
        "hybrid_close_scatter",
        "benchmarks/rebound_hybrid_comparison.py",
        ("--profile", "smoke"),
        "hybrid_smoke.json",
        "jx.rebound_hybrid.close_scatter.v1",
        "jx.rebound_hybrid.all_active_close_scatter.v1",
        None,
        170_407,
        "f6bf49a5041b63a50ec59d4a909ef490ccfae334ceba2cb82e7846a2c38a13b0",
        frozenset(
            {
                "all_far_control",
                "benchmark_id",
                "claim_controls",
                "content_integrity",
                "headline_lanes",
                "interpretation",
                "limitations",
                "method_provenance",
                "numerical_references",
                "problem",
                "profile",
                "refinement",
                "regression_gates",
                "runtime_provenance",
                "schema",
                "serialization",
                "timing_disclosure",
            }
        ),
    ),
)

_LANE_CONTRACTS = {
    "leapfrog_binary": {
        "jx_kdk": (
            "JX_PRIMARY_FIXED_GRID",
            "integrator.symplectic.kdk_leapfrog_2",
            False,
        ),
        "jx_rkf78_adaptive_reference": (
            "JX_ADAPTIVE_NUMERICAL_REFERENCE_NOT_TRUTH",
            "integrator.adaptive.rkf78.fehlberg_1968",
            False,
        ),
        "rebound_leapfrog": (
            "OPTIONAL_EXTERNAL_COMPARATOR",
            "rebound.integrator.leapfrog.5.1.1",
            True,
        ),
    },
    "whfast_weak_three_body": {
        "jx_wh_long_p64": (
            "JX_PRIMARY_FIXED_GRID",
            "integrator.symplectic.wisdom_holman_jacobi_kdk_2",
            False,
        ),
        "jx_wh_short_p128": (
            "JX_PRIMARY_FIXED_GRID",
            "integrator.symplectic.wisdom_holman_jacobi_kdk_2",
            False,
        ),
        "jx_wh_short_p64": (
            "JX_PRIMARY_FIXED_GRID",
            "integrator.symplectic.wisdom_holman_jacobi_kdk_2",
            False,
        ),
        "rebound_whfast_long_p64": (
            "OPTIONAL_EXTERNAL_COMPARATOR",
            "rebound.integrator.whfast.5.1.1",
            True,
        ),
        "rebound_whfast_short_p128": (
            "OPTIONAL_EXTERNAL_COMPARATOR",
            "rebound.integrator.whfast.5.1.1",
            True,
        ),
        "rebound_whfast_short_p64": (
            "OPTIONAL_EXTERNAL_COMPARATOR",
            "rebound.integrator.whfast.5.1.1",
            True,
        ),
    },
    "hybrid_close_scatter": {
        "jx_hybrid_p256": (
            "JX_PRIMARY_GUARDED_HYBRID",
            "integrator.hybrid.wisdom_holman_jacobi_rkf78_cartesian.v1",
            False,
        ),
        "rebound_mercurius_p256": (
            "OPTIONAL_EXTERNAL_COMPARATOR",
            "rebound.integrator.mercurius.5.1.1",
            True,
        ),
        "rebound_trace_p256": (
            "OPTIONAL_EXTERNAL_COMPARATOR",
            "rebound.integrator.trace.5.1.1",
            True,
        ),
    },
}

_WHFAST_PHASE_REFERENCES = {
    "jx_wh_long_p64": "rebound_ias15_long_declared_reference",
    "jx_wh_short_p128": "rebound_ias15_short_declared_reference",
    "jx_wh_short_p64": "rebound_ias15_short_declared_reference",
    "rebound_whfast_long_p64": (
        "rebound_ias15_long_whfast_p64_observed_reference"
    ),
    "rebound_whfast_short_p128": (
        "rebound_ias15_short_whfast_p128_observed_reference"
    ),
    "rebound_whfast_short_p64": (
        "rebound_ias15_short_whfast_p64_observed_reference"
    ),
}

_HYBRID_PHASE_REFERENCES = {
    "jx_hybrid_p256": "rebound_ias15_jx_exact_labels_dt_p128",
    "rebound_mercurius_p256": "rebound_ias15_actual_clock_p256",
    "rebound_trace_p256": "rebound_ias15_actual_clock_p256",
}

_BUNDLE_FILENAMES = tuple(
    sorted((*tuple(study.report_filename for study in _STUDIES), REPORT_FILENAME))
)
_COMPLETE_BUNDLE_FILENAMES = frozenset((*_BUNDLE_FILENAMES, CHECKSUM_FILENAME))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _domain_sha256(domain: str, value: Any) -> str:
    return _sha256(
        domain.encode("ascii") + b"\0" + _canonical_bytes(_typed_digest_tree(value))
    )


def _typed_digest_tree(value: Any) -> Any:
    """Return an injective, fully type-tagged representation of a JSON tree."""

    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        if value.bit_length() > MAX_JSON_INTEGER_BITS:
            raise ChallengerError("digest input integer exceeds the bit-length cap")
        return ["int", str(value)]
    if type(value) is float:
        if not math.isfinite(value):
            raise ChallengerError("digest input contains a nonfinite binary64 value")
        return ["binary64", value.hex()]
    if type(value) is str:
        return ["string", value]
    if type(value) is list:
        return ["list", [_typed_digest_tree(item) for item in value]]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ChallengerError("digest input contains a non-string mapping key")
        return [
            "object",
            [[key, _typed_digest_tree(value[key])] for key in sorted(value)],
        ]
    raise ChallengerError("digest input contains an unsupported value type")


def _canonical_bytes(value: Any) -> bytes:
    _validate_json_tree(value)
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ChallengerError("value cannot be encoded as finite canonical JSON") from exc
    return text.encode("ascii")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ChallengerError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_int(raw: str) -> int:
    if len(raw) > 1_235:
        raise ChallengerError("JSON integer exceeds the lexical cap")
    value = int(raw, 10)
    if value.bit_length() > MAX_JSON_INTEGER_BITS:
        raise ChallengerError("JSON integer exceeds the bit-length cap")
    return value


def _parse_float(raw: str) -> float:
    if len(raw) > 128:
        raise ChallengerError("JSON float exceeds the lexical cap")
    value = float(raw)
    if not math.isfinite(value):
        raise ChallengerError("JSON contains a nonfinite number")
    return value


def _reject_constant(raw: str) -> Any:
    raise ChallengerError(f"JSON contains forbidden constant {raw!r}")


def _parse_json_document(data: bytes) -> dict[str, Any]:
    if len(data) > MAX_CHILD_STDOUT_BYTES:
        raise ChallengerError("JSON document exceeds the byte cap")
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_parse_int,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except ChallengerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ChallengerError("child output is not one strict UTF-8 JSON document") from exc
    _validate_json_tree(value)
    if type(value) is not dict:
        raise ChallengerError("JSON document root must be an object")
    return value


def _validate_json_tree(value: Any) -> None:
    nodes = 0
    string_bytes = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ChallengerError("JSON tree exceeds the node cap")
        if depth > MAX_JSON_DEPTH:
            raise ChallengerError("JSON tree exceeds the depth cap")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            if item.bit_length() > MAX_JSON_INTEGER_BITS:
                raise ChallengerError("JSON integer exceeds the bit-length cap")
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ChallengerError("JSON tree contains a nonfinite float")
            continue
        if type(item) is str:
            string_bytes += len(item.encode("utf-8"))
            if string_bytes > MAX_JSON_STRING_BYTES:
                raise ChallengerError("JSON tree exceeds the cumulative string cap")
            continue
        if type(item) is list:
            for child in reversed(item):
                stack.append((child, depth + 1))
            continue
        if type(item) is dict:
            for key, child in reversed(tuple(item.items())):
                if type(key) is not str:
                    raise ChallengerError("JSON object key is not an exact string")
                string_bytes += len(key.encode("utf-8"))
                if string_bytes > MAX_JSON_STRING_BYTES:
                    raise ChallengerError("JSON tree exceeds the cumulative string cap")
                stack.append((child, depth + 1))
            continue
        raise ChallengerError(
            f"JSON tree contains unsupported value type {type(item).__name__}"
        )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ChallengerError(f"{label} must be an exact object")
    return value


def _exact_string(value: Any, label: str) -> str:
    if type(value) is not str or not value or len(value) > 1024:
        raise ChallengerError(f"{label} must be a bounded nonempty exact string")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ChallengerError(f"{label} must be an exact bool")
    return value


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ChallengerError(f"{label} must be an exact integer >= {minimum}")
    return value


def _finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ChallengerError(f"{label} must be an exact finite float")
    return value


def _sha(value: Any, label: str) -> str:
    text = _exact_string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ChallengerError(f"{label} must be one lowercase SHA-256 digest")
    return text


def _all_false_claims(value: Any, label: str) -> dict[str, bool]:
    claims = _mapping(value, label)
    if not claims:
        raise ChallengerError(f"{label} cannot be empty")
    for key, claim in claims.items():
        _exact_string(key, f"{label} key")
        if _exact_bool(claim, f"{label}.{key}") is not False:
            raise ChallengerError(f"{label}.{key} must remain false")
    return claims


def _path(value: Any, keys: Iterable[str], label: str) -> Any:
    current = value
    for key in keys:
        current = _mapping(current, label)
        if key not in current:
            raise ChallengerError(f"{label} lacks required field {key!r}")
        current = current[key]
    return current


def _file_identity(path: Path, relative_path: str) -> dict[str, Any]:
    try:
        link_status = os.lstat(path)
    except OSError as exc:
        raise ChallengerError(f"required source is unavailable: {relative_path}") from exc
    if stat.S_ISLNK(link_status.st_mode) or not stat.S_ISREG(link_status.st_mode):
        raise ChallengerError(f"required source is not a regular non-symlink: {relative_path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ChallengerError(f"required source is not regular: {relative_path}")
        if before.st_size < 1 or before.st_size > MAX_SOURCE_BYTES:
            raise ChallengerError(f"required source violates the size cap: {relative_path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ChallengerError(f"required source was truncated: {relative_path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ChallengerError(f"required source grew while read: {relative_path}")
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ChallengerError(f"required source cannot be read: {relative_path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise ChallengerError(f"required source changed while read: {relative_path}")
    payload = b"".join(chunks)
    return {
        "relative_path": relative_path,
        "size_bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _source_roster(repository_root: Path) -> tuple[dict[str, Any], ...]:
    paths = ("benchmarks/jx_challenger_v1.py",) + tuple(
        study.script_relative_path for study in _STUDIES
    )
    return tuple(
        _file_identity(repository_root / relative, relative) for relative in paths
    )


def _source_by_path(roster: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in roster:
        relative = _exact_string(item.get("relative_path"), "source relative_path")
        if relative in result:
            raise ChallengerError("source roster contains a duplicate path")
        _exact_int(item.get("size_bytes"), "source size_bytes", minimum=1)
        _sha(item.get("sha256"), "source sha256")
        result[relative] = item
    return result


def _clean_child_environment() -> dict[str, str]:
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TZ": "UTC",
    }


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 0.5
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired as exc:
        raise ChallengerError("child process group could not be reaped") from exc


def _run_bounded_child(
    command: tuple[str, ...], *, repository_root: Path, timeout_seconds: float
) -> tuple[bytes, bytes, int, float]:
    if type(timeout_seconds) is not float or not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ChallengerError("child timeout must be a positive exact finite float")
    start = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    try:
        process = subprocess.Popen(
            command,
            cwd=repository_root,
            env=_clean_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )
        if process.stdout is None or process.stderr is None:
            raise ChallengerError("child pipe construction failed")
        for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, label)
        deadline = start + timeout_seconds
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise ChallengerError("child process exceeded the wall-time cap")
            for key, _ in selector.select(min(remaining, 0.25)):
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                target = stdout if key.data == "stdout" else stderr
                cap = (
                    MAX_CHILD_STDOUT_BYTES
                    if key.data == "stdout"
                    else MAX_CHILD_STDERR_BYTES
                )
                if len(target) + len(chunk) > cap:
                    raise ChallengerError(f"child {key.data} exceeded the byte cap")
                target.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise ChallengerError("child process exceeded the wall-time cap")
        return_code = process.wait(timeout=remaining)
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None:
            _terminate_process_group(process)
        raise ChallengerError("child process execution failed") from exc
    except BaseException:
        if process is not None:
            _terminate_process_group(process)
        raise
    finally:
        selector.close()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
    elapsed = time.monotonic() - start
    if return_code != 0:
        detail = bytes(stderr).decode("utf-8", errors="replace").strip()
        if len(detail) > 2000:
            detail = detail[:2000] + "..."
        suffix = f": {detail}" if detail else ""
        raise ChallengerError(f"child process exited with status {return_code}{suffix}")
    if stderr:
        raise ChallengerError("successful child emitted unexpected stderr bytes")
    return bytes(stdout), bytes(stderr), return_code, elapsed


def _child_command(
    repository_root: Path, study: StudySpec, rebound_mode: str
) -> tuple[str, ...]:
    script = (repository_root / study.script_relative_path).resolve(strict=True)
    source_root = (repository_root / "src").resolve(strict=True)
    return (
        sys.executable,
        "-I",
        "-B",
        "-c",
        _BOOTSTRAP,
        os.fspath(source_root),
        os.fspath(script),
        *study.child_arguments,
        "--rebound-mode",
        rebound_mode,
    )


def _child_script_identity(report: dict[str, Any], study: StudySpec) -> dict[str, Any]:
    if study.study_id == "leapfrog_binary":
        value = _path(report, ("runtime", "benchmark_script"), "leapfrog runtime")
    else:
        value = _path(
            report,
            ("runtime_provenance", "jx", "benchmark_script"),
            f"{study.study_id} runtime",
        )
    identity = _mapping(value, f"{study.study_id} benchmark source")
    return {
        "relative_path": study.script_relative_path,
        "size_bytes": _exact_int(
            identity.get("size_bytes"), f"{study.study_id} source size", minimum=1
        ),
        "sha256": _sha(identity.get("sha256"), f"{study.study_id} source sha256"),
    }


def _support_script_identity(report: dict[str, Any], study: StudySpec) -> dict[str, Any]:
    value = _path(
        report,
        ("runtime_provenance", "jx", "provenance_support_script"),
        f"{study.study_id} provenance support",
    )
    identity = _mapping(value, f"{study.study_id} provenance support")
    return {
        "relative_path": "benchmarks/rebound_leapfrog_comparison.py",
        "size_bytes": _exact_int(
            identity.get("size_bytes"),
            f"{study.study_id} provenance support size",
            minimum=1,
        ),
        "sha256": _sha(
            identity.get("sha256"),
            f"{study.study_id} provenance support sha256",
        ),
    }


def _engine_tree_sha256(report: dict[str, Any], study: StudySpec) -> str:
    if study.study_id == "leapfrog_binary":
        value = _path(report, ("runtime", "engine_source_tree_sha256"), "engine source")
    else:
        value = _path(
            report,
            ("runtime_provenance", "jx", "engine_sources", "tree_sha256"),
            "engine source",
        )
    return _sha(value, f"{study.study_id} engine source tree sha256")


def _external_status(
    report: dict[str, Any], study: StudySpec, rebound_mode: str
) -> dict[str, Any]:
    if study.study_id == "leapfrog_binary":
        lane = _mapping(_path(report, ("lanes", "rebound_leapfrog"), "external lane"), "external lane")
        executed = _exact_bool(lane.get("available"), "rebound_leapfrog.available")
        reason_raw = lane.get("unavailability_reason") if not executed else None
        if rebound_mode == "disabled":
            if executed or reason_raw != "disabled_by_cli":
                raise ChallengerError("disabled leapfrog study has inconsistent external status")
            reason = "EXPLICITLY_DISABLED"
        elif executed:
            reason = "EXECUTED_EXACT_REBOUND_5_1_1"
        else:
            _exact_string(reason_raw, "leapfrog unavailability reason")
            reason = "EXACT_REBOUND_5_1_1_UNAVAILABLE"
    else:
        status = _mapping(
            _path(report, ("runtime_provenance", "external_status"), "external status"),
            "external status",
        )
        if status.get("requested_mode") != rebound_mode:
            raise ChallengerError(f"{study.study_id} changed the requested REBOUND mode")
        if status.get("required_version") != REQUIRED_REBOUND_VERSION:
            raise ChallengerError(f"{study.study_id} changed the required REBOUND version")
        if status.get("authority_authorized") is not False:
            raise ChallengerError(f"{study.study_id} external status claimed authority")
        executed = _exact_bool(
            status.get("external_lanes_executed"),
            f"{study.study_id} external_lanes_executed",
        )
        raw_reason = _exact_string(
            status.get("reason"), f"{study.study_id} external reason"
        )
        expected_raw_reason = (
            "EXACT_REBOUND_5_1_1_LOADED"
            if executed
            else (
                "EXPLICITLY_DISABLED"
                if rebound_mode == "disabled"
                else "EXACT_REBOUND_5_1_1_UNAVAILABLE"
            )
        )
        if raw_reason != expected_raw_reason:
            raise ChallengerError(f"{study.study_id} external reason is inconsistent")
        reason = (
            "EXECUTED_EXACT_REBOUND_5_1_1" if executed else raw_reason
        )
    if rebound_mode == "required" and not executed:
        raise ChallengerError("required REBOUND mode returned no external execution")
    return {
        "requested_mode": rebound_mode,
        "required_version": REQUIRED_REBOUND_VERSION,
        "external_lanes_executed": executed,
        "reason": reason,
        "authority_authorized": False,
    }


def _digest_item(scope: str, value: Any, label: str) -> dict[str, str]:
    return {"scope": scope, "sha256": _sha(value, label)}


def _lane_digest(summary: dict[str, Any]) -> dict[str, Any]:
    core = copy.deepcopy(summary)
    core.pop("summary_sha256", None)
    summary["summary_sha256"] = _domain_sha256(_LANE_DOMAIN, core)
    return summary


def _leapfrog_lane(lane_id: str, lane: dict[str, Any], *, executed: bool) -> dict[str, Any]:
    if not executed:
        if lane_id != "rebound_leapfrog":
            raise ChallengerError("only the external leapfrog lane may be unavailable")
        return _lane_digest(
            {
                "engine_id": lane_id,
                "method_id": None,
                "role": "OPTIONAL_EXTERNAL_COMPARATOR",
                "executed": False,
                "native_digests": [],
                "invariant_baseline": None,
                "relative_total_energy_max": None,
                "relative_angular_momentum_max": None,
                "position_vector_l2_max": None,
                "velocity_vector_l2_max": None,
                "phase_metric": None,
                "work_accounting": None,
            }
        )
    accounting = _mapping(lane.get("accounting"), f"{lane_id} accounting")
    invariants = _mapping(lane.get("invariant_envelopes"), f"{lane_id} invariants")
    oracle = _mapping(lane.get("declared_grid_oracle_errors"), f"{lane_id} oracle")
    position = _mapping(oracle.get("position"), f"{lane_id} position")
    velocity = _mapping(oracle.get("velocity"), f"{lane_id} velocity")
    phase = _mapping(oracle.get("phase"), f"{lane_id} phase")
    digests = [
        _digest_item("trajectory", lane.get("trajectory_sha256"), f"{lane_id} trajectory")
    ]
    if lane_id == "jx_kdk":
        digests.extend(
            (
                _digest_item("result", accounting.get("result_content_sha256"), "KDK result"),
                _digest_item("schedule", accounting.get("schedule_content_sha256"), "KDK schedule"),
            )
        )
        role = "JX_PRIMARY_FIXED_GRID"
    elif lane_id == "jx_rkf78_adaptive_reference":
        ledger = _mapping(accounting.get("accepted_step_ledger"), "RKF78 ledger")
        digests.append(
            _digest_item(
                "accepted_step_ledger",
                ledger.get("content_integrity_sha256"),
                "RKF78 accepted-step ledger",
            )
        )
        role = "JX_ADAPTIVE_NUMERICAL_REFERENCE_NOT_TRUTH"
    else:
        role = "OPTIONAL_EXTERNAL_COMPARATOR"
    method_id = (
        "rebound.integrator.leapfrog.5.1.1"
        if lane_id == "rebound_leapfrog"
        else _exact_string(accounting.get("method_id"), f"{lane_id} method_id")
    )
    phase_metric = {
        "observable": "RELATIVE_BINARY_ORBIT_ANGLE_IN_XY_PLANE",
        "reference_id": _exact_string(phase.get("reference_label"), f"{lane_id} phase reference"),
        "sign_convention": "CANDIDATE_MINUS_ANALYTIC_REFERENCE_UNWRAPPED",
        "maximum_abs_radians": _finite_float(phase.get("maximum_abs_radians"), f"{lane_id} phase max"),
        "rms_radians": _finite_float(phase.get("rms_radians"), f"{lane_id} phase rms"),
        "final_signed_radians": _finite_float(phase.get("final_signed_radians"), f"{lane_id} phase final"),
    }
    external_force_exposed: bool | None = None
    if lane_id == "rebound_leapfrog":
        external_force_exposed = _exact_bool(
            accounting.get("force_evaluations_exposed"),
            f"{lane_id} force exposure",
        )
        if external_force_exposed is not False:
            raise ChallengerError("REBOUND leapfrog cannot expose unreported force work")
    return _lane_digest(
        {
            "engine_id": lane_id,
            "method_id": method_id,
            "role": role,
            "executed": True,
            "native_digests": digests,
            "invariant_baseline": (
                "ANALYTIC_FIXTURE_INITIAL_ENERGY_AND_ANGULAR_MOMENTUM"
            ),
            "relative_total_energy_max": _finite_float(
                invariants.get("relative_total_energy_max"), f"{lane_id} energy"
            ),
            "relative_angular_momentum_max": _finite_float(
                invariants.get("relative_angular_momentum_max"), f"{lane_id} angular momentum"
            ),
            "position_vector_l2_max": _finite_float(
                position.get("vector_l2_max"), f"{lane_id} position error"
            ),
            "velocity_vector_l2_max": _finite_float(
                velocity.get("vector_l2_max"), f"{lane_id} velocity error"
            ),
            "phase_metric": phase_metric,
            "work_accounting": (
                {
                    "completed_steps": _exact_int(
                        accounting.get("completed_integer_steps"),
                        f"{lane_id} completed steps",
                        minimum=1,
                    ),
                    "force_evaluations": None,
                    "force_evaluations_exposed": external_force_exposed,
                }
                if lane_id == "rebound_leapfrog"
                else {
                    "completed_steps": _exact_int(
                        accounting.get("completed_steps", accounting.get("accepted_steps")),
                        f"{lane_id} completed steps",
                        minimum=1,
                    ),
                    "force_evaluations": _exact_int(
                        accounting.get("force_evaluations"),
                        f"{lane_id} force evaluations",
                        minimum=1,
                    ),
                    "force_evaluations_exposed": True,
                }
            ),
        }
    )


def _whfast_lane(lane_id: str, lane: dict[str, Any]) -> dict[str, Any]:
    accounting = _mapping(lane.get("accounting"), f"{lane_id} accounting")
    invariants = _mapping(
        lane.get("invariant_checkpoint_samples"), f"{lane_id} invariants"
    )
    integrity = _mapping(lane.get("lane_content_integrity"), f"{lane_id} integrity")
    phase_source = lane.get("observed_epoch_phase_proxy")
    state_source = lane.get("observed_epoch_state_error")
    if (phase_source is None) != (state_source is None):
        raise ChallengerError(f"{lane_id} phase/state reference availability differs")
    phase_metric: dict[str, Any] | None
    position_max: float | None
    velocity_max: float | None
    if phase_source is None:
        phase_metric = None
        position_max = None
        velocity_max = None
    else:
        phase = _mapping(phase_source, f"{lane_id} phase proxy")
        state = _mapping(state_source, f"{lane_id} state error")
        position = _mapping(state.get("position"), f"{lane_id} position error")
        velocity = _mapping(state.get("velocity"), f"{lane_id} velocity error")
        bodies = _mapping(phase.get("bodies"), f"{lane_id} phase bodies")
        if set(bodies) != {"INNER", "OUTER"}:
            raise ChallengerError(f"{lane_id} phase body roster changed")
        phase_metric = {
            "observable": _exact_string(phase.get("definition"), f"{lane_id} phase definition"),
            "reference_id": _exact_string(
                state.get("reference_engine_id"), f"{lane_id} reference engine"
            ),
            "sign_convention": "CANDIDATE_MINUS_IAS15_REFERENCE_WRAPPED_PER_BODY",
            "maximum_abs_radians": _finite_float(
                phase.get("max_abs_radians"), f"{lane_id} phase max"
            ),
            "rms_radians": _finite_float(phase.get("rms_radians"), f"{lane_id} phase rms"),
            "body_metrics": copy.deepcopy(bodies),
        }
        position_max = _finite_float(position.get("vector_l2_max"), f"{lane_id} position")
        velocity_max = _finite_float(velocity.get("vector_l2_max"), f"{lane_id} velocity")
    external = not lane_id.startswith("jx_wh_")
    role = (
        "JX_PRIMARY_FIXED_GRID"
        if not external
        else "OPTIONAL_EXTERNAL_COMPARATOR"
    )
    method_id = (
        "rebound.integrator.whfast.5.1.1"
        if external
        else _exact_string(accounting.get("method_id"), f"{lane_id} method_id")
    )
    work_accounting = (
        {
            "completed_steps": _exact_int(
                accounting.get("completed_integer_steps"),
                f"{lane_id} completed steps",
                minimum=1,
            ),
            "primary_force_evaluations": None,
            "force_evaluations_exposed": False,
        }
        if external
        else {
            "completed_steps": _exact_int(
                accounting.get("completed_steps"), f"{lane_id} completed steps", minimum=1
            ),
            "primary_force_evaluations": _exact_int(
                accounting.get("primary_map_force_evaluations"),
                f"{lane_id} primary force evaluations",
                minimum=1,
            ),
            "force_evaluations_exposed": True,
        }
    )
    return _lane_digest(
        {
            "engine_id": lane_id,
            "method_id": method_id,
            "role": role,
            "executed": True,
            "native_digests": [
                _digest_item("lane_content", integrity.get("sha256"), f"{lane_id} lane digest")
            ],
            "invariant_baseline": "EACH_LANE_FIRST_RETAINED_STATE",
            "relative_total_energy_max": _finite_float(
                invariants.get("relative_total_energy_max"), f"{lane_id} energy"
            ),
            "relative_angular_momentum_max": _finite_float(
                invariants.get("relative_angular_momentum_max"), f"{lane_id} angular momentum"
            ),
            "position_vector_l2_max": position_max,
            "velocity_vector_l2_max": velocity_max,
            "phase_metric": phase_metric,
            "work_accounting": work_accounting,
        }
    )


def _hybrid_lane(lane_id: str, lane: dict[str, Any]) -> dict[str, Any]:
    invariants = _mapping(lane.get("headline_invariants"), f"{lane_id} invariants")
    accuracy_source = lane.get("headline_accuracy_against_numerical_reference")
    phase_metric: dict[str, Any] | None
    position_max: float | None
    velocity_max: float | None
    if accuracy_source is None:
        phase_metric = None
        position_max = None
        velocity_max = None
    else:
        accuracy = _mapping(accuracy_source, f"{lane_id} accuracy")
        phase = _mapping(
            accuracy.get("inner_outer_relative_vector_phase_proxy"),
            f"{lane_id} phase proxy",
        )
        position = _mapping(accuracy.get("position"), f"{lane_id} position error")
        velocity = _mapping(accuracy.get("velocity"), f"{lane_id} velocity error")
        phase_metric = {
            "observable": _exact_string(phase.get("proxy"), f"{lane_id} phase proxy"),
            "reference_id": (
                "rebound_ias15_jx_exact_labels_dt_p128"
                if lane_id.startswith("jx_hybrid_")
                else f"rebound_ias15_actual_clock_p{lane_id.rsplit('_p', 1)[1]}"
            ),
            "sign_convention": _exact_string(
                phase.get("sign_convention"), f"{lane_id} phase sign convention"
            ),
            "maximum_abs_radians": _finite_float(
                phase.get("maximum_abs_radians"), f"{lane_id} phase max"
            ),
            "rms_radians": _finite_float(phase.get("rms_radians"), f"{lane_id} phase rms"),
            "final_signed_radians": _finite_float(
                phase.get("final_signed_radians"), f"{lane_id} phase final"
            ),
        }
        position_max = _finite_float(position.get("vector_l2_max"), f"{lane_id} position")
        velocity_max = _finite_float(velocity.get("vector_l2_max"), f"{lane_id} velocity")
    accounting = _mapping(lane.get("accounting"), f"{lane_id} accounting")
    role = (
        "JX_PRIMARY_GUARDED_HYBRID"
        if lane_id.startswith("jx_hybrid_")
        else "OPTIONAL_EXTERNAL_COMPARATOR"
    )
    digest_fields = (
        ("lane_content", "lane_content_sha256"),
        ("declared_epochs", "declared_epochs_content_sha256"),
        ("observed_epochs", "observed_epochs_content_sha256"),
        ("positions", "positions_content_sha256"),
        ("velocities", "velocities_content_sha256"),
    )
    external = not lane_id.startswith("jx_hybrid_")
    work_accounting = (
        {
            "completed_steps": _exact_int(
                lane.get("outer_step_or_target_interval_count"),
                f"{lane_id} completed steps",
                minimum=1,
            ),
            "mode_counts": None,
            "internal_force_evaluations_exposed": False,
        }
        if external
        else {
            "completed_steps": _exact_int(
                lane.get("outer_step_or_target_interval_count"),
                f"{lane_id} completed steps",
                minimum=1,
            ),
            "mode_counts": copy.deepcopy(
                _mapping(accounting.get("mode_counts"), f"{lane_id} mode counts")
            ),
            "internal_force_evaluations_exposed": True,
        }
    )
    return _lane_digest(
        {
            "engine_id": lane_id,
            "method_id": _exact_string(lane.get("method_id"), f"{lane_id} method_id"),
            "role": role,
            "executed": True,
            "native_digests": [
                _digest_item(scope, lane.get(field), f"{lane_id} {scope} digest")
                for scope, field in digest_fields
            ],
            "invariant_baseline": "EACH_LANE_FIRST_RETAINED_STATE",
            "relative_total_energy_max": _finite_float(
                invariants.get("relative_total_energy_max"), f"{lane_id} energy"
            ),
            "relative_angular_momentum_max": _finite_float(
                invariants.get("relative_angular_momentum_max"), f"{lane_id} angular momentum"
            ),
            "position_vector_l2_max": position_max,
            "velocity_vector_l2_max": velocity_max,
            "phase_metric": phase_metric,
            "work_accounting": work_accounting,
        }
    )


def _validate_child_header(report: dict[str, Any], study: StudySpec) -> None:
    if set(report) != set(study.exact_top_keys):
        raise ChallengerError(f"{study.study_id} top-level report schema changed")
    if report.get("schema") != study.child_schema:
        raise ChallengerError(f"{study.study_id} child schema changed")
    if report.get("benchmark_id") != study.child_benchmark_id:
        raise ChallengerError(f"{study.study_id} benchmark identifier changed")
    profile = _mapping(report.get("profile"), f"{study.study_id} profile")
    if profile.get("name") != PROFILE_NAME or profile.get("named_full_profile") is not False:
        raise ChallengerError(f"{study.study_id} is not the exact smoke profile")
    actual_profile_id = profile.get("profile_id")
    if actual_profile_id != study.child_profile_id:
        raise ChallengerError(f"{study.study_id} profile identifier changed")
    if study.study_id == "hybrid_close_scatter" and (
        profile.get("include_limitations") is not False
        or profile.get("jx_divisors") != [256]
        or profile.get("external_divisors") != [256]
        or profile.get("base_steps_at_p256") != 128
    ):
        raise ChallengerError("hybrid smoke profile expanded its exact lane scope")
    if study.study_id == "leapfrog_binary" and (
        profile.get("periods") != 1
        or profile.get("samples_per_period") != 16
        or profile.get("fixed_steps_per_period") != 256
    ):
        raise ChallengerError("leapfrog smoke workload changed")
    if study.study_id == "whfast_weak_three_body" and (
        profile.get("short_periods") != 1
        or profile.get("long_periods") != 2
        or profile.get("samples_per_period") != 4
    ):
        raise ChallengerError("WHFast smoke workload changed")
    _all_false_claims(report.get("claim_controls"), f"{study.study_id} claim controls")


def _build_study_summary(
    report: dict[str, Any],
    study: StudySpec,
    rebound_mode: str,
    source_identity: dict[str, Any],
    support_identity: dict[str, Any],
) -> dict[str, Any]:
    _validate_child_header(report, study)
    reported_source = _child_script_identity(report, study)
    if reported_source != source_identity:
        raise ChallengerError(f"{study.study_id} source identity does not match its report")
    status = _external_status(report, study, rebound_mode)
    if study.study_id == "leapfrog_binary":
        raw_lanes = _mapping(report.get("lanes"), "leapfrog lanes")
        if set(raw_lanes) != {
            "jx_kdk",
            "jx_rkf78_adaptive_reference",
            "rebound_leapfrog",
        }:
            raise ChallengerError("leapfrog lane roster changed")
        lanes = {
            lane_id: _leapfrog_lane(
                lane_id,
                _mapping(raw_lanes[lane_id], f"{lane_id} lane"),
                executed=(lane_id != "rebound_leapfrog" or status["external_lanes_executed"]),
            )
            for lane_id in sorted(raw_lanes)
        }
        transitive: list[dict[str, Any]] = []
        reference_digests: dict[str, str] = {}
        native_report_semantic = None
    elif study.study_id == "whfast_weak_three_body":
        if _support_script_identity(report, study) != support_identity:
            raise ChallengerError("WHFast provenance-support source identity changed")
        raw_lanes = _mapping(report.get("lanes"), "WHFast lanes")
        expected = {
            "jx_wh_long_p64",
            "jx_wh_short_p128",
            "jx_wh_short_p64",
        }
        if status["external_lanes_executed"]:
            expected |= {
                "rebound_whfast_long_p64",
                "rebound_whfast_short_p128",
                "rebound_whfast_short_p64",
            }
        if set(raw_lanes) != expected:
            raise ChallengerError("WHFast lane roster changed")
        lanes = {
            lane_id: _whfast_lane(lane_id, _mapping(raw_lanes[lane_id], f"{lane_id} lane"))
            for lane_id in sorted(raw_lanes)
        }
        raw_references = _mapping(
            report.get("ias15_numerical_references"), "WHFast IAS15 references"
        )
        if status["external_lanes_executed"]:
            expected_references = {
                "rebound_ias15_short_declared_reference",
                "rebound_ias15_long_declared_reference",
                "rebound_ias15_short_whfast_p64_observed_reference",
                "rebound_ias15_short_whfast_p128_observed_reference",
                "rebound_ias15_long_whfast_p64_observed_reference",
            }
            if set(raw_references) != expected_references:
                raise ChallengerError("WHFast IAS15 numerical-reference roster changed")
        elif raw_references:
            raise ChallengerError("unexecuted WHFast study retained IAS15 references")
        reference_digests = {
            reference_id: _sha(
                _path(
                    raw_references[reference_id],
                    ("lane_content_integrity", "sha256"),
                    f"{reference_id} lane content",
                ),
                f"{reference_id} lane content sha256",
            )
            for reference_id in sorted(raw_references)
        }
        transitive = [copy.deepcopy(support_identity)]
        native_report_semantic = None
    else:
        if _support_script_identity(report, study) != support_identity:
            raise ChallengerError("hybrid provenance-support source identity changed")
        raw_lanes = _mapping(report.get("headline_lanes"), "hybrid headline lanes")
        expected = {"jx_hybrid_p256"}
        if status["external_lanes_executed"]:
            expected |= {"rebound_mercurius_p256", "rebound_trace_p256"}
        if set(raw_lanes) != expected:
            raise ChallengerError("hybrid headline lane roster changed")
        lanes = {
            lane_id: _hybrid_lane(lane_id, _mapping(raw_lanes[lane_id], f"{lane_id} lane"))
            for lane_id in sorted(raw_lanes)
        }
        raw_references = _mapping(
            report.get("numerical_references"), "hybrid numerical references"
        )
        expected_references = (
            {
                "rebound_ias15_jx_exact_labels_dt_p128",
                "rebound_ias15_jx_exact_labels_dt_p1024_sensitivity",
                "rebound_ias15_actual_clock_p256",
            }
            if status["external_lanes_executed"]
            else set()
        )
        if set(raw_references) != expected_references:
            raise ChallengerError("hybrid numerical-reference roster changed")
        reference_digests = {
            reference_id: _sha(
                _path(
                    raw_references[reference_id],
                    ("lane_content_sha256",),
                    f"{reference_id} lane content",
                ),
                f"{reference_id} lane content sha256",
            )
            for reference_id in sorted(raw_references)
        }
        transitive = [copy.deepcopy(support_identity)]
        native_report_semantic = _sha(
            _path(
                report,
                ("content_integrity", "report_semantic_content_sha256"),
                "hybrid report semantic digest",
            ),
            "hybrid report semantic digest",
        )
    summary: dict[str, Any] = {
        "study_id": study.study_id,
        "child_schema": study.child_schema,
        "child_benchmark_id": study.child_benchmark_id,
        "child_profile_id": study.child_profile_id,
        "child_source": copy.deepcopy(source_identity),
        "transitive_source_identities": transitive,
        "external_status": status,
        "lanes": lanes,
        "native_report_semantic_content_sha256": native_report_semantic,
        "numerical_reference_content_sha256": reference_digests,
        "jx_lane_ids": [
            lane_id for lane_id, lane in lanes.items() if lane["role"].startswith("JX_")
        ],
        "external_lane_ids": [
            lane_id
            for lane_id, lane in lanes.items()
            if lane["role"] == "OPTIONAL_EXTERNAL_COMPARATOR" and lane["executed"]
        ],
    }
    summary["study_summary_semantic_content_sha256"] = _domain_sha256(
        _STUDY_DOMAIN, summary
    )
    return summary


def _semantic_core(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "benchmark_id": report["benchmark_id"],
        "profile": copy.deepcopy(report["profile"]),
        "runner_source": copy.deepcopy(report["runner_source"]),
        "jx_engine_source_tree_sha256": report["jx_engine_source_tree_sha256"],
        "studies": copy.deepcopy(report["studies"]),
        "claim_controls": copy.deepcopy(report["claim_controls"]),
        "serialization": copy.deepcopy(report["serialization"]),
    }


def _attach_integrity(report: dict[str, Any]) -> dict[str, Any]:
    if "content_integrity" in report:
        raise ChallengerError("content integrity must be attached exactly once")
    semantic_sha = _domain_sha256(_SEMANTIC_DOMAIN, _semantic_core(report))
    integrity_without_content = {
        "classification": "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY",
        "authority_authorized": False,
        "semantic_content_domain": _SEMANTIC_DOMAIN,
        "semantic_content_sha256": semantic_sha,
        "semantic_content_excludes_exactly": ["execution_diagnostics", "content_integrity"],
        "aggregate_content_domain": _CONTENT_DOMAIN,
    }
    content_core = copy.deepcopy(report)
    content_core["content_integrity"] = copy.deepcopy(integrity_without_content)
    content_sha = _domain_sha256(_CONTENT_DOMAIN, content_core)
    report["content_integrity"] = {
        **integrity_without_content,
        "aggregate_content_sha256": content_sha,
    }
    return report


def _validate_compact_phase(
    study_id: str,
    lane_id: str,
    value: Any,
    *,
    executed: bool,
    external_enabled: bool,
) -> None:
    phase_expected = executed and (
        study_id == "leapfrog_binary" or external_enabled
    )
    if not phase_expected:
        if value is not None:
            raise ChallengerError(f"{lane_id} invented a phase comparison")
        return
    phase = _mapping(value, f"{lane_id} phase metric")
    if study_id == "whfast_weak_three_body":
        if set(phase) != {
            "observable",
            "reference_id",
            "sign_convention",
            "maximum_abs_radians",
            "rms_radians",
            "body_metrics",
        }:
            raise ChallengerError(f"{lane_id} phase schema changed")
        if (
            phase.get("observable")
            != "WRAPPED_PLANAR_POLAR_ANGLE_OF_BODY_MINUS_PRIMARY_"
            "CANDIDATE_MINUS_IAS15_REFERENCE"
            or phase.get("reference_id") != _WHFAST_PHASE_REFERENCES[lane_id]
            or phase.get("sign_convention")
            != "CANDIDATE_MINUS_IAS15_REFERENCE_WRAPPED_PER_BODY"
        ):
            raise ChallengerError(f"{lane_id} phase meaning changed")
        bodies = _mapping(phase.get("body_metrics"), f"{lane_id} body phase metrics")
        if set(bodies) != {"INNER", "OUTER"}:
            raise ChallengerError(f"{lane_id} phase body roster changed")
        for body_id, metrics in bodies.items():
            metric_map = _mapping(metrics, f"{lane_id} {body_id} phase metrics")
            if set(metric_map) != {
                "max_abs_radians",
                "rms_radians",
                "final_wrapped_radians",
            }:
                raise ChallengerError(f"{lane_id} phase body schema changed")
            if (
                _finite_float(
                    metric_map.get("max_abs_radians"),
                    f"{lane_id} {body_id} phase maximum",
                )
                < 0.0
                or _finite_float(
                    metric_map.get("rms_radians"),
                    f"{lane_id} {body_id} phase RMS",
                )
                < 0.0
            ):
                raise ChallengerError(f"{lane_id} phase body norm is negative")
            _finite_float(
                metric_map.get("final_wrapped_radians"),
                f"{lane_id} {body_id} final phase",
            )
    else:
        if set(phase) != {
            "observable",
            "reference_id",
            "sign_convention",
            "maximum_abs_radians",
            "rms_radians",
            "final_signed_radians",
        }:
            raise ChallengerError(f"{lane_id} phase schema changed")
        if study_id == "leapfrog_binary":
            expected_observable = "RELATIVE_BINARY_ORBIT_ANGLE_IN_XY_PLANE"
            expected_reference = "analytic_declared_grid_orbit"
            expected_sign = "CANDIDATE_MINUS_ANALYTIC_REFERENCE_UNWRAPPED"
        else:
            expected_observable = (
                "INNER_TO_OUTER_RELATIVE_POSITION_VECTOR_ANGLE_IN_XY_PLANE"
            )
            expected_reference = _HYBRID_PHASE_REFERENCES[lane_id]
            expected_sign = (
                "ATAN2_CROSS_CANDIDATE_REFERENCE_REFERENCE_MINUS_CANDIDATE"
            )
        if (
            phase.get("observable") != expected_observable
            or phase.get("reference_id") != expected_reference
            or phase.get("sign_convention") != expected_sign
        ):
            raise ChallengerError(f"{lane_id} phase meaning changed")
    if (
        _finite_float(
            phase.get("maximum_abs_radians"), f"{lane_id} phase maximum"
        )
        < 0.0
        or _finite_float(phase.get("rms_radians"), f"{lane_id} phase RMS")
        < 0.0
    ):
        raise ChallengerError(f"{lane_id} phase norm is negative")
    if study_id != "whfast_weak_three_body":
        _finite_float(
            phase.get("final_signed_radians"), f"{lane_id} final phase"
        )


def _validate_compact_work(
    study_id: str,
    lane_id: str,
    value: Any,
    *,
    executed: bool,
    optional_external: bool,
) -> None:
    if not executed:
        if value is not None:
            raise ChallengerError(f"{lane_id} invented work accounting")
        return
    work = _mapping(value, f"{lane_id} work accounting")
    completed = _exact_int(
        work.get("completed_steps"), f"{lane_id} completed steps", minimum=1
    )
    if study_id == "leapfrog_binary":
        if set(work) != {
            "completed_steps",
            "force_evaluations",
            "force_evaluations_exposed",
        }:
            raise ChallengerError(f"{lane_id} work-accounting schema changed")
        exposed = _exact_bool(
            work.get("force_evaluations_exposed"), f"{lane_id} force exposure"
        )
        if optional_external:
            if exposed or work.get("force_evaluations") is not None:
                raise ChallengerError(f"{lane_id} invented external force work")
        elif not exposed:
            raise ChallengerError(f"{lane_id} hid JX force work")
        else:
            force_evaluations = _exact_int(
                work.get("force_evaluations"),
                f"{lane_id} force evaluations",
                minimum=1,
            )
            if lane_id == "jx_kdk" and force_evaluations != completed + 1:
                raise ChallengerError(f"{lane_id} JX work accounting changed")
    elif study_id == "whfast_weak_three_body":
        if set(work) != {
            "completed_steps",
            "primary_force_evaluations",
            "force_evaluations_exposed",
        }:
            raise ChallengerError(f"{lane_id} work-accounting schema changed")
        exposed = _exact_bool(
            work.get("force_evaluations_exposed"), f"{lane_id} force exposure"
        )
        if optional_external:
            if exposed or work.get("primary_force_evaluations") is not None:
                raise ChallengerError(f"{lane_id} invented external force work")
        else:
            primary = _exact_int(
                work.get("primary_force_evaluations"),
                f"{lane_id} primary force evaluations",
                minimum=1,
            )
            if not exposed or primary != completed + 1:
                raise ChallengerError(f"{lane_id} JX work accounting changed")
    else:
        if set(work) != {
            "completed_steps",
            "mode_counts",
            "internal_force_evaluations_exposed",
        }:
            raise ChallengerError(f"{lane_id} work-accounting schema changed")
        exposed = _exact_bool(
            work.get("internal_force_evaluations_exposed"),
            f"{lane_id} internal-force exposure",
        )
        if optional_external:
            if exposed or work.get("mode_counts") is not None:
                raise ChallengerError(f"{lane_id} invented external hybrid work")
        else:
            counts = _mapping(work.get("mode_counts"), f"{lane_id} mode counts")
            if set(counts) != {
                "CARTESIAN_RKF78_NEAR_FULL_INTERVAL",
                "WISDOM_HOLMAN_FAR",
            }:
                raise ChallengerError(f"{lane_id} mode-count roster changed")
            values = tuple(
                _exact_int(count, f"{lane_id} {mode} count")
                for mode, count in counts.items()
            )
            if not exposed or sum(values) != completed:
                raise ChallengerError(f"{lane_id} hybrid work accounting changed")


def validate_challenger_report(report: Any) -> None:
    root = _mapping(report, "Challenger report")
    expected_keys = {
        "schema",
        "benchmark_id",
        "profile",
        "runner_source",
        "jx_engine_source_tree_sha256",
        "studies",
        "claim_controls",
        "serialization",
        "execution_diagnostics",
        "content_integrity",
    }
    if set(root) != expected_keys:
        raise ChallengerError("Challenger report top-level schema changed")
    if root.get("schema") != SCHEMA or root.get("benchmark_id") != BENCHMARK_ID:
        raise ChallengerError("Challenger report identity changed")
    profile = _mapping(root.get("profile"), "Challenger profile")
    if set(profile) != {
        "name",
        "rebound_mode",
        "study_order",
        "child_process_isolation",
    }:
        raise ChallengerError("Challenger profile schema changed")
    mode = profile.get("rebound_mode")
    if type(mode) is not str or mode not in REBOUND_MODES:
        raise ChallengerError("Challenger REBOUND mode is invalid")
    expected_order = [study.study_id for study in _STUDIES]
    if (
        profile.get("name") != PROFILE_NAME
        or profile.get("study_order") != expected_order
        or profile.get("child_process_isolation") is not True
    ):
        raise ChallengerError("Challenger profile changed")
    runner_source = _mapping(root.get("runner_source"), "runner source")
    if runner_source.get("relative_path") != "benchmarks/jx_challenger_v1.py":
        raise ChallengerError("runner source path changed")
    _exact_int(runner_source.get("size_bytes"), "runner source size", minimum=1)
    _sha(runner_source.get("sha256"), "runner source sha256")
    if root.get("jx_engine_source_tree_sha256") != EXPECTED_ENGINE_SOURCE_TREE_SHA256:
        raise ChallengerError("JX engine source tree differs from frozen Challenger V1")
    studies = root.get("studies")
    if type(studies) is not list or len(studies) != len(_STUDIES):
        raise ChallengerError("Challenger study roster changed")
    external_states: list[bool] = []
    for expected, summary in zip(_STUDIES, studies):
        item = _mapping(summary, f"{expected.study_id} summary")
        if set(item) != {
            "study_id",
            "child_schema",
            "child_benchmark_id",
            "child_profile_id",
            "child_source",
            "transitive_source_identities",
            "external_status",
            "lanes",
            "native_report_semantic_content_sha256",
            "numerical_reference_content_sha256",
            "jx_lane_ids",
            "external_lane_ids",
            "study_summary_semantic_content_sha256",
        }:
            raise ChallengerError(f"{expected.study_id} summary schema changed")
        if item.get("study_id") != expected.study_id:
            raise ChallengerError("Challenger study ordering changed")
        if item.get("child_schema") != expected.child_schema:
            raise ChallengerError(f"{expected.study_id} child schema changed")
        if item.get("child_benchmark_id") != expected.child_benchmark_id:
            raise ChallengerError(f"{expected.study_id} benchmark identifier changed")
        if item.get("child_profile_id") != expected.child_profile_id:
            raise ChallengerError(f"{expected.study_id} profile identifier changed")
        child_source = _mapping(item.get("child_source"), "child source identity")
        if set(child_source) != {"relative_path", "size_bytes", "sha256"}:
            raise ChallengerError("child source identity schema changed")
        if child_source.get("relative_path") != expected.script_relative_path:
            raise ChallengerError("child source relative path changed")
        if (
            child_source.get("size_bytes") != expected.script_size_bytes
            or child_source.get("sha256") != expected.script_sha256
        ):
            raise ChallengerError("child source differs from the frozen V1 source")
        transitive = item.get("transitive_source_identities")
        if type(transitive) is not list or len(transitive) != (
            0 if expected.study_id == "leapfrog_binary" else 1
        ):
            raise ChallengerError("transitive source identity roster changed")
        if transitive:
            support = _mapping(transitive[0], "transitive source identity")
            if (
                set(support) != {"relative_path", "size_bytes", "sha256"}
                or support.get("relative_path")
                != "benchmarks/rebound_leapfrog_comparison.py"
            ):
                raise ChallengerError("transitive source identity schema changed")
            frozen_support = _STUDIES[0]
            if (
                support.get("size_bytes") != frozen_support.script_size_bytes
                or support.get("sha256") != frozen_support.script_sha256
            ):
                raise ChallengerError(
                    "transitive support differs from the frozen V1 source"
                )
        status = _mapping(item.get("external_status"), f"{expected.study_id} external status")
        if set(status) != {
            "requested_mode",
            "required_version",
            "external_lanes_executed",
            "reason",
            "authority_authorized",
        }:
            raise ChallengerError("normalized external-status schema changed")
        if (
            status.get("required_version") != REQUIRED_REBOUND_VERSION
            or status.get("authority_authorized") is not False
        ):
            raise ChallengerError("normalized external identity or authority changed")
        external_states.append(
            _exact_bool(
                status.get("external_lanes_executed"),
                f"{expected.study_id} external execution",
            )
        )
        if status.get("requested_mode") != mode:
            raise ChallengerError("study REBOUND mode differs from aggregate")
        expected_reason = (
            "EXECUTED_EXACT_REBOUND_5_1_1"
            if external_states[-1]
            else (
                "EXPLICITLY_DISABLED"
                if mode == "disabled"
                else "EXACT_REBOUND_5_1_1_UNAVAILABLE"
            )
        )
        if status.get("reason") != expected_reason:
            raise ChallengerError("normalized external reason changed")
        lanes = _mapping(item.get("lanes"), f"{expected.study_id} lanes")
        if not lanes or len(lanes) > 16:
            raise ChallengerError(f"{expected.study_id} lane count is invalid")
        lane_contracts = _LANE_CONTRACTS[expected.study_id]
        expected_lane_ids = {
            lane_id
            for lane_id, (_, _, optional_external) in lane_contracts.items()
            if not optional_external
            or external_states[-1]
            or expected.study_id == "leapfrog_binary"
        }
        if set(lanes) != expected_lane_ids:
            raise ChallengerError(f"{expected.study_id} compact lane roster changed")
        for lane_id, lane in lanes.items():
            lane_map = _mapping(lane, f"{lane_id} lane summary")
            if set(lane_map) != {
                "engine_id",
                "method_id",
                "role",
                "executed",
                "native_digests",
                "invariant_baseline",
                "relative_total_energy_max",
                "relative_angular_momentum_max",
                "position_vector_l2_max",
                "velocity_vector_l2_max",
                "phase_metric",
                "work_accounting",
                "summary_sha256",
            }:
                raise ChallengerError(f"{lane_id} lane summary schema changed")
            if lane_map.get("engine_id") != lane_id:
                raise ChallengerError(f"{lane_id} lane identity changed")
            expected_role, expected_method, optional_external = lane_contracts[lane_id]
            role = lane_map.get("role")
            if role != expected_role:
                raise ChallengerError(f"{lane_id} lane role changed")
            executed = _exact_bool(lane_map.get("executed"), f"{lane_id} executed")
            expected_executed = not optional_external or external_states[-1]
            if executed is not expected_executed:
                raise ChallengerError(f"{lane_id} execution status changed")
            if executed:
                expected_baseline = (
                    "ANALYTIC_FIXTURE_INITIAL_ENERGY_AND_ANGULAR_MOMENTUM"
                    if expected.study_id == "leapfrog_binary"
                    else "EACH_LANE_FIRST_RETAINED_STATE"
                )
                if lane_map.get("invariant_baseline") != expected_baseline:
                    raise ChallengerError(f"{lane_id} invariant baseline changed")
                if lane_map.get("method_id") != expected_method:
                    raise ChallengerError(f"{lane_id} method identity changed")
                digests = lane_map.get("native_digests")
                if type(digests) is not list or not 1 <= len(digests) <= 8:
                    raise ChallengerError(f"{lane_id} native digest roster changed")
                for digest in digests:
                    digest_map = _mapping(digest, f"{lane_id} native digest")
                    if set(digest_map) != {"scope", "sha256"}:
                        raise ChallengerError(f"{lane_id} native digest schema changed")
                    _exact_string(digest_map.get("scope"), f"{lane_id} digest scope")
                    _sha(digest_map.get("sha256"), f"{lane_id} native digest")
                if expected.study_id == "leapfrog_binary":
                    expected_scopes = (
                        ("trajectory", "result", "schedule")
                        if lane_id == "jx_kdk"
                        else (
                            ("trajectory", "accepted_step_ledger")
                            if lane_id == "jx_rkf78_adaptive_reference"
                            else ("trajectory",)
                        )
                    )
                elif expected.study_id == "whfast_weak_three_body":
                    expected_scopes = ("lane_content",)
                else:
                    expected_scopes = (
                        "lane_content",
                        "declared_epochs",
                        "observed_epochs",
                        "positions",
                        "velocities",
                    )
                if tuple(item["scope"] for item in digests) != expected_scopes:
                    raise ChallengerError(f"{lane_id} native digest scope changed")
                for metric_name in (
                    "relative_total_energy_max",
                    "relative_angular_momentum_max",
                ):
                    if _finite_float(lane_map.get(metric_name), f"{lane_id} {metric_name}") < 0.0:
                        raise ChallengerError(f"{lane_id} {metric_name} cannot be negative")
                _mapping(lane_map.get("work_accounting"), f"{lane_id} work accounting")
            elif (
                role != "OPTIONAL_EXTERNAL_COMPARATOR"
                or lane_map.get("method_id") is not None
                or lane_map.get("native_digests") != []
                or lane_map.get("invariant_baseline") is not None
                or lane_map.get("relative_total_energy_max") is not None
                or lane_map.get("relative_angular_momentum_max") is not None
                or lane_map.get("work_accounting") is not None
            ):
                raise ChallengerError("unexecuted external lane retained execution evidence")
            state_metrics_expected = executed and (
                expected.study_id == "leapfrog_binary" or external_states[-1]
            )
            for metric_name in (
                "position_vector_l2_max",
                "velocity_vector_l2_max",
            ):
                metric = lane_map.get(metric_name)
                if state_metrics_expected:
                    if _finite_float(metric, f"{lane_id} {metric_name}") < 0.0:
                        raise ChallengerError(f"{lane_id} {metric_name} is negative")
                elif metric is not None:
                    raise ChallengerError(
                        f"{lane_id} invented an unavailable state-error metric"
                    )
            _validate_compact_phase(
                expected.study_id,
                lane_id,
                lane_map.get("phase_metric"),
                executed=executed,
                external_enabled=external_states[-1],
            )
            _validate_compact_work(
                expected.study_id,
                lane_id,
                lane_map.get("work_accounting"),
                executed=executed,
                optional_external=optional_external,
            )
            claimed = _sha(lane_map.get("summary_sha256"), f"{lane_id} summary digest")
            lane_core = copy.deepcopy(lane_map)
            del lane_core["summary_sha256"]
            if claimed != _domain_sha256(_LANE_DOMAIN, lane_core):
                raise ChallengerError(f"{lane_id} lane summary seal is stale")
        claimed_study = _sha(
            item.get("study_summary_semantic_content_sha256"),
            f"{expected.study_id} study summary digest",
        )
        study_core = copy.deepcopy(item)
        del study_core["study_summary_semantic_content_sha256"]
        if claimed_study != _domain_sha256(_STUDY_DOMAIN, study_core):
            raise ChallengerError(f"{expected.study_id} study summary seal is stale")
        reference_digests = _mapping(
            item.get("numerical_reference_content_sha256"),
            f"{expected.study_id} numerical-reference digests",
        )
        if len(reference_digests) > 16:
            raise ChallengerError("numerical-reference digest roster exceeds the cap")
        for reference_id, digest in reference_digests.items():
            _exact_string(reference_id, "numerical-reference id")
            _sha(digest, "numerical-reference content sha256")
        if expected.study_id == "whfast_weak_three_body":
            expected_reference_ids = (
                {
                    "rebound_ias15_short_declared_reference",
                    "rebound_ias15_long_declared_reference",
                    "rebound_ias15_short_whfast_p64_observed_reference",
                    "rebound_ias15_short_whfast_p128_observed_reference",
                    "rebound_ias15_long_whfast_p64_observed_reference",
                }
                if external_states[-1]
                else set()
            )
            if set(reference_digests) != expected_reference_ids:
                raise ChallengerError("WHFast compact numerical-reference roster changed")
        elif expected.study_id == "hybrid_close_scatter":
            expected_reference_ids = (
                {
                    "rebound_ias15_jx_exact_labels_dt_p128",
                    "rebound_ias15_jx_exact_labels_dt_p1024_sensitivity",
                    "rebound_ias15_actual_clock_p256",
                }
                if external_states[-1]
                else set()
            )
            if set(reference_digests) != expected_reference_ids:
                raise ChallengerError("hybrid compact numerical-reference roster changed")
        elif reference_digests:
            raise ChallengerError("study retained an unexpected numerical-reference digest map")
        native_report = item.get("native_report_semantic_content_sha256")
        if expected.study_id == "hybrid_close_scatter":
            _sha(native_report, "hybrid native report semantic sha256")
        elif native_report is not None:
            raise ChallengerError("non-hybrid study invented a report-semantic digest")
        expected_jx = sorted(
            lane_id
            for lane_id, (_, _, optional_external) in lane_contracts.items()
            if not optional_external
        )
        expected_external = sorted(
            lane_id
            for lane_id, (_, _, optional_external) in lane_contracts.items()
            if optional_external and external_states[-1]
        )
        if item.get("jx_lane_ids") != expected_jx or item.get("external_lane_ids") != expected_external:
            raise ChallengerError("compact lane identifier roster is stale")
    if len(set(external_states)) != 1:
        raise ChallengerError("the three studies disagree on external execution availability")
    if mode == "disabled" and external_states[0]:
        raise ChallengerError("disabled mode executed external lanes")
    if mode == "required" and not external_states[0]:
        raise ChallengerError("required mode did not execute external lanes")
    if root.get("claim_controls") != _AGGREGATE_CLAIM_CONTROLS:
        raise ChallengerError("aggregate claim controls changed")
    serialization = _mapping(root.get("serialization"), "serialization")
    if serialization != {
        "format": "SORTED_INDENTED_ASCII_FINITE_JSON",
        "raw_child_reports_preserved_verbatim": True,
        "raw_child_reports_include_timing_and_path_diagnostics": True,
        "deterministic_pretty_report_bytes_claimed": False,
        "semantic_digest_excludes_only_execution_diagnostics_and_integrity": True,
        "digest_tree_encoding": "FULLY_TYPE_TAGGED_JSON_TREE_WITH_BINARY64_HEX",
        "resource_caps": {
            "child_stdout_bytes": MAX_CHILD_STDOUT_BYTES,
            "child_stderr_bytes": MAX_CHILD_STDERR_BYTES,
            "json_depth": MAX_JSON_DEPTH,
            "json_nodes": MAX_JSON_NODES,
            "json_cumulative_string_bytes": MAX_JSON_STRING_BYTES,
            "json_integer_bits": MAX_JSON_INTEGER_BITS,
        },
    }:
        raise ChallengerError("serialization declaration changed")
    diagnostics = _mapping(root.get("execution_diagnostics"), "execution diagnostics")
    if set(diagnostics) != {
        "completed",
        "python_executable",
        "child_process_model",
        "children",
        "timeout_seconds_per_child",
        "process_independence_or_security_sandbox_claimed",
    }:
        raise ChallengerError("execution diagnostics schema changed")
    if (
        diagnostics.get("child_process_model")
        != "DISTINCT_CPYTHON_PROCESS_PER_EXISTING_COMPARATOR"
        or diagnostics.get("process_independence_or_security_sandbox_claimed")
        is not False
        or diagnostics.get("timeout_seconds_per_child") != CHILD_TIMEOUT_SECONDS
    ):
        raise ChallengerError("execution process declaration changed")
    _exact_string(diagnostics.get("python_executable"), "Python executable diagnostic")
    records = diagnostics.get("children")
    if type(records) is not list or len(records) != len(_STUDIES):
        raise ChallengerError("execution diagnostic roster changed")
    for expected, item in zip(_STUDIES, records):
        record = _mapping(item, f"{expected.study_id} diagnostics")
        if set(record) != {
            "study_id",
            "report_filename",
            "return_code",
            "wall_seconds",
            "stdout_size_bytes",
            "stdout_sha256",
            "stderr_size_bytes",
            "stderr_sha256",
        }:
            raise ChallengerError("child execution diagnostic schema changed")
        if record.get("study_id") != expected.study_id or record.get("report_filename") != expected.report_filename:
            raise ChallengerError("execution diagnostic ordering changed")
        stdout_size = _exact_int(record.get("stdout_size_bytes"), "child stdout size", minimum=1)
        if stdout_size > MAX_CHILD_STDOUT_BYTES:
            raise ChallengerError("child stdout diagnostic exceeds the cap")
        _sha(record.get("stdout_sha256"), "child stdout sha256")
        stderr_size = _exact_int(record.get("stderr_size_bytes"), "child stderr size")
        if stderr_size != 0:
            raise ChallengerError("successful child retained stderr bytes")
        if record.get("stderr_sha256") != _sha256(b""):
            raise ChallengerError("successful child stderr digest is not the empty digest")
        if _finite_float(record.get("wall_seconds"), "child wall seconds") < 0.0:
            raise ChallengerError("child wall time cannot be negative")
        if type(record.get("return_code")) is not int or record.get("return_code") != 0:
            raise ChallengerError("completed report records a failed child")
    if diagnostics.get("completed") is not True:
        raise ChallengerError("Challenger report is not complete")
    integrity = _mapping(root.get("content_integrity"), "content integrity")
    if set(integrity) != {
        "classification",
        "authority_authorized",
        "semantic_content_domain",
        "semantic_content_sha256",
        "semantic_content_excludes_exactly",
        "aggregate_content_domain",
        "aggregate_content_sha256",
    } or (
        integrity.get("classification") != "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"
        or integrity.get("authority_authorized") is not False
        or integrity.get("semantic_content_domain") != _SEMANTIC_DOMAIN
        or integrity.get("semantic_content_excludes_exactly")
        != ["execution_diagnostics", "content_integrity"]
        or integrity.get("aggregate_content_domain") != _CONTENT_DOMAIN
    ):
        raise ChallengerError("content-integrity declaration changed")
    claimed_semantic = _sha(
        integrity.get("semantic_content_sha256"), "semantic content sha256"
    )
    if claimed_semantic != _domain_sha256(_SEMANTIC_DOMAIN, _semantic_core(root)):
        raise ChallengerError("Challenger semantic content seal is stale")
    claimed_content = _sha(
        integrity.get("aggregate_content_sha256"), "aggregate content sha256"
    )
    content_core = copy.deepcopy(root)
    del content_core["content_integrity"]["aggregate_content_sha256"]
    if claimed_content != _domain_sha256(_CONTENT_DOMAIN, content_core):
        raise ChallengerError("Challenger aggregate content seal is stale")
    _validate_json_tree(root)


def _build_report(
    *,
    rebound_mode: str,
    source_roster: tuple[dict[str, Any], ...],
    engine_tree_sha256: str,
    studies: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    sources = _source_by_path(source_roster)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "profile": {
            "name": PROFILE_NAME,
            "rebound_mode": rebound_mode,
            "study_order": [study.study_id for study in _STUDIES],
            "child_process_isolation": True,
        },
        "runner_source": copy.deepcopy(sources["benchmarks/jx_challenger_v1.py"]),
        "jx_engine_source_tree_sha256": engine_tree_sha256,
        "studies": studies,
        "claim_controls": copy.deepcopy(_AGGREGATE_CLAIM_CONTROLS),
        "serialization": {
            "format": "SORTED_INDENTED_ASCII_FINITE_JSON",
            "raw_child_reports_preserved_verbatim": True,
            "raw_child_reports_include_timing_and_path_diagnostics": True,
            "deterministic_pretty_report_bytes_claimed": False,
            "semantic_digest_excludes_only_execution_diagnostics_and_integrity": True,
            "digest_tree_encoding": "FULLY_TYPE_TAGGED_JSON_TREE_WITH_BINARY64_HEX",
            "resource_caps": {
                "child_stdout_bytes": MAX_CHILD_STDOUT_BYTES,
                "child_stderr_bytes": MAX_CHILD_STDERR_BYTES,
                "json_depth": MAX_JSON_DEPTH,
                "json_nodes": MAX_JSON_NODES,
                "json_cumulative_string_bytes": MAX_JSON_STRING_BYTES,
                "json_integer_bits": MAX_JSON_INTEGER_BITS,
            },
        },
        "execution_diagnostics": {
            "completed": True,
            "python_executable": sys.executable,
            "child_process_model": "DISTINCT_CPYTHON_PROCESS_PER_EXISTING_COMPARATOR",
            "children": diagnostics,
            "timeout_seconds_per_child": CHILD_TIMEOUT_SECONDS,
            "process_independence_or_security_sandbox_claimed": False,
        },
    }
    _attach_integrity(report)
    validate_challenger_report(report)
    return report


def _render_report(report: dict[str, Any]) -> bytes:
    validate_challenger_report(report)
    return (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_file(path: Path, payload: bytes) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ChallengerError(f"staged file write stalled: {path.name!r}")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise ChallengerError(f"cannot publish staged file {path.name!r}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    try:
        status = os.lstat(path)
    except OSError as exc:
        raise ChallengerError(f"bundle file is unavailable: {path.name}") from exc
    if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise ChallengerError(f"bundle member is not a regular file: {path.name}")
    if status.st_size < 1 or status.st_size > maximum_bytes:
        raise ChallengerError(f"bundle member violates its size cap: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != status.st_size
            or before.st_dev != status.st_dev
            or before.st_ino != status.st_ino
        ):
            raise ChallengerError(f"bundle member changed before read: {path.name}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                raise ChallengerError(f"bundle member was truncated: {path.name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ChallengerError(f"bundle member grew while read: {path.name}")
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ChallengerError(f"bundle member cannot be read: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise ChallengerError(f"bundle member changed while read: {path.name}")
    payload = b"".join(chunks)
    if len(payload) != status.st_size or len(payload) > maximum_bytes:
        raise ChallengerError(f"bundle member violates its stable size: {path.name}")
    return payload


def _checksum_payload(payloads: dict[str, bytes]) -> bytes:
    if set(payloads) != set(_BUNDLE_FILENAMES):
        raise ChallengerError("checksum payload roster changed")
    return "".join(
        f"{_sha256(payloads[name])}  {name}\n" for name in sorted(payloads)
    ).encode("ascii")


def verify_bundle_directory(directory: os.PathLike[str] | str) -> dict[str, Any]:
    root = Path(directory)
    try:
        status = os.lstat(root)
        with os.scandir(root) as entries:
            names = {entry.name for entry in entries}
    except OSError as exc:
        raise ChallengerError("bundle directory cannot be inspected") from exc
    if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise ChallengerError("bundle root must be a real directory")
    if names != _COMPLETE_BUNDLE_FILENAMES:
        raise ChallengerError("bundle file roster changed")
    payloads = {
        name: _read_regular_file(root / name, MAX_CHILD_STDOUT_BYTES)
        for name in _BUNDLE_FILENAMES
    }
    checksum = _read_regular_file(root / CHECKSUM_FILENAME, 4096)
    if checksum != _checksum_payload(payloads):
        raise ChallengerError("bundle checksum manifest is stale")
    aggregate = _parse_json_document(payloads[REPORT_FILENAME])
    validate_challenger_report(aggregate)
    diagnostics = _mapping(aggregate["execution_diagnostics"], "execution diagnostics")
    diagnostic_by_id = {
        _exact_string(item.get("study_id"), "diagnostic study id"): item
        for item in diagnostics["children"]
    }
    source_map = {
        item["study_id"]: _mapping(item["child_source"], "child source")
        for item in aggregate["studies"]
    }
    support = next(
        item["child_source"]
        for item in aggregate["studies"]
        if item["study_id"] == "leapfrog_binary"
    )
    rebuilt: list[dict[str, Any]] = []
    for spec in _STUDIES:
        raw = payloads[spec.report_filename]
        diagnostic = _mapping(diagnostic_by_id.get(spec.study_id), "child diagnostic")
        if (
            diagnostic.get("stdout_size_bytes") != len(raw)
            or diagnostic.get("stdout_sha256") != _sha256(raw)
        ):
            raise ChallengerError(f"{spec.study_id} raw report does not match diagnostics")
        child = _parse_json_document(raw)
        rebuilt.append(
            _build_study_summary(
                child,
                spec,
                aggregate["profile"]["rebound_mode"],
                source_map[spec.study_id],
                support,
            )
        )
    if rebuilt != aggregate["studies"]:
        raise ChallengerError("aggregate study summaries do not match raw child reports")
    return aggregate


def _resolve_new_output_directory(output_directory: Path) -> tuple[Path, Path]:
    target_input = Path(os.path.abspath(os.fspath(output_directory)))
    if not target_input.name:
        raise ChallengerError("output directory cannot be a filesystem root")
    try:
        parent = target_input.parent.resolve(strict=True)
        parent_status = os.lstat(parent)
    except OSError as exc:
        raise ChallengerError("output parent directory must already exist") from exc
    if not stat.S_ISDIR(parent_status.st_mode) or stat.S_ISLNK(parent_status.st_mode):
        raise ChallengerError("output parent must be a real directory")
    target = parent / target_input.name
    if os.path.lexists(target):
        raise ChallengerError("output directory already exists")
    return parent, target


def _publish_bundle(
    output_directory: Path, payloads: dict[str, bytes], report: dict[str, Any]
) -> None:
    parent, target = _resolve_new_output_directory(output_directory)
    try:
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=parent))
    except OSError as exc:
        raise ChallengerError("cannot create the private bundle stage") from exc
    published = False
    renamed = False
    try:
        stage_identity = os.lstat(stage)
    except OSError as exc:
        raise ChallengerError("cannot observe the private bundle stage") from exc
    try:
        try:
            os.chmod(stage, 0o700)
            aggregate_bytes = _render_report(report)
            complete_payloads = dict(payloads)
            complete_payloads[REPORT_FILENAME] = aggregate_bytes
            if set(complete_payloads) != set(_BUNDLE_FILENAMES):
                raise ChallengerError("bundle payload roster changed")
            for name in _BUNDLE_FILENAMES:
                _write_file(stage / name, complete_payloads[name])
            _write_file(stage / CHECKSUM_FILENAME, _checksum_payload(complete_payloads))
            directory_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            verify_bundle_directory(stage)
            if os.path.lexists(target):
                raise ChallengerError("output directory appeared before publication")
            os.rename(stage, target)
            renamed = True
            parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            verify_bundle_directory(target)
            published = True
        except OSError as exc:
            raise ChallengerError("bundle publication failed") from exc
    finally:
        cleanup_target: Path | None = None
        if not published and os.path.lexists(stage):
            cleanup_target = stage
        elif not published and renamed and os.path.lexists(target):
            cleanup_target = target
        if cleanup_target is not None:
            try:
                cleanup_identity = os.lstat(cleanup_target)
                if (
                    cleanup_identity.st_dev != stage_identity.st_dev
                    or cleanup_identity.st_ino != stage_identity.st_ino
                    or not stat.S_ISDIR(cleanup_identity.st_mode)
                ):
                    raise ChallengerError(
                        "refusing to clean a replaced bundle-stage directory"
                    )
                shutil.rmtree(cleanup_target)
            except OSError as exc:
                raise ChallengerError("private bundle-stage cleanup failed") from exc


def run_challenger(
    output_directory: os.PathLike[str] | str,
    *,
    rebound_mode: str = "disabled",
) -> dict[str, Any]:
    """Execute and atomically publish the locked Challenger V1 smoke bundle."""

    if type(rebound_mode) is not str or rebound_mode not in REBOUND_MODES:
        raise ChallengerError(f"rebound_mode must be one of {REBOUND_MODES!r}")
    if type(output_directory) is not str and not isinstance(output_directory, Path):
        raise ChallengerError("output_directory must be a str or pathlib.Path")
    _resolve_new_output_directory(Path(output_directory))
    repository_root = Path(__file__).resolve().parents[1]
    before = _source_roster(repository_root)
    sources = _source_by_path(before)
    for spec in _STUDIES:
        identity = sources[spec.script_relative_path]
        if (
            identity["size_bytes"] != spec.script_size_bytes
            or identity["sha256"] != spec.script_sha256
        ):
            raise ChallengerError(
                f"{spec.study_id} source differs from frozen Challenger V1"
            )
    payloads: dict[str, bytes] = {}
    summaries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    engine_hashes: list[str] = []
    support = sources["benchmarks/rebound_leapfrog_comparison.py"]
    for spec in _STUDIES:
        command = _child_command(repository_root, spec, rebound_mode)
        stdout, stderr, return_code, elapsed = _run_bounded_child(
            command,
            repository_root=repository_root,
            timeout_seconds=CHILD_TIMEOUT_SECONDS,
        )
        child = _parse_json_document(stdout)
        source = sources[spec.script_relative_path]
        summaries.append(
            _build_study_summary(child, spec, rebound_mode, source, support)
        )
        engine_hashes.append(_engine_tree_sha256(child, spec))
        payloads[spec.report_filename] = stdout
        diagnostics.append(
            {
                "study_id": spec.study_id,
                "report_filename": spec.report_filename,
                "return_code": return_code,
                "wall_seconds": float(elapsed),
                "stdout_size_bytes": len(stdout),
                "stdout_sha256": _sha256(stdout),
                "stderr_size_bytes": len(stderr),
                "stderr_sha256": _sha256(stderr),
            }
        )
    if len(set(engine_hashes)) != 1:
        raise ChallengerError("child reports disagree on the JX engine source tree")
    if engine_hashes[0] != EXPECTED_ENGINE_SOURCE_TREE_SHA256:
        raise ChallengerError("JX engine source tree differs from frozen Challenger V1")
    if _source_roster(repository_root) != before:
        raise ChallengerError("Challenger or comparator source changed during execution")
    report = _build_report(
        rebound_mode=rebound_mode,
        source_roster=before,
        engine_tree_sha256=engine_hashes[0],
        studies=summaries,
        diagnostics=diagnostics,
    )
    _publish_bundle(Path(output_directory), payloads, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked smoke-only JX Challenger bundle in three isolated "
            "child processes; no ranking or superiority claim is produced."
        )
    )
    parser.add_argument(
        "--rebound-mode",
        choices=REBOUND_MODES,
        default="disabled",
        help="disabled is dependency-free; auto/required accept exact REBOUND 5.1.1 only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory to receive the complete five-file bundle",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        report = run_challenger(
            arguments.output_dir,
            rebound_mode=arguments.rebound_mode,
        )
    except ChallengerError as exc:
        print(f"{parser.prog}: execution failed: {exc}", file=sys.stderr)
        return 1
    digest = report["content_integrity"]["semantic_content_sha256"]
    print(f"published {arguments.output_dir} semantic_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
