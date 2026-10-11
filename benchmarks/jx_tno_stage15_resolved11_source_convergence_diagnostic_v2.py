#!/usr/bin/env python3
"""V2 successor for the resolved-11 source-convergence diagnostic.

V2 executes the frozen V1 implementation as an exact, hash-bound dependency,
changes every local schema/domain/output address to V2, and replaces only the
V1 IAS15 annual-step adjacency validator.  The V1 attempt completed all of its
source-only integrations but published no result because strict zip received
the complete annual roster and its one-element-short tail.  V2 validates the
same 1,001 entries through 1,000 indexed adjacent pairs.  No physics,
integration, gate, disposition, or claim-ceiling rule is otherwise changed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_SOURCE_PATH = (
    PROJECT_ROOT / "benchmarks"
    / "jx_tno_stage15_resolved11_source_convergence_diagnostic_v1.py"
)
V1_TEST_PATH = (
    PROJECT_ROOT / "tests"
    / "test_tno_stage15_resolved11_source_convergence_diagnostic_v1.py"
)
V1_ROOT = (
    PROJECT_ROOT / "runs"
    / "tno_stage15_resolved11_source_convergence_diagnostic_v1"
)
V1_PROTOCOL_PATH = V1_ROOT / "PROTOCOL.json"

V1_SOURCE_BYTES = 145_758
V1_SOURCE_SHA256 = "fef7d1c585566acbcb67ab78f177303d8d55ab13a738b6c1f730a7f2bda291e4"
V1_TEST_BYTES = 29_777
V1_TEST_SHA256 = "2a618175e3c41e5e599918bc8770807ae655f3d02373826090698c664e68c025"
V1_PROTOCOL_BYTES = 12_387
V1_PROTOCOL_SHA256 = "75ca8e4781e36b2e32312fdf57fe3c11a450376070098b12c9c73acfeb52d44e"
V1_PROTOCOL_CONTENT_SHA256 = "5c62c4c776b02a374dda2381ef20123af70f3d212264d8f20e92a4137c20b267"
V1_PROTOCOL_SCHEMA = "jx.tno.stage15-resolved11-source-convergence-diagnostic.protocol.v1"
V1_PROTOCOL_DOMAIN = b"jx.tno.stage15-resolved11-source-convergence-diagnostic.protocol.v1\0"
V1_GIT_BRANCH = "research/tno-stage15-resolved11-source-convergence-v1-preregistered-20261010"
V1_GIT_COMMIT = "9da26171a080062bda5cdd064d723ef4df20ba9d"
V1_GIT_TAG = "tno-stage15-resolved11-source-convergence-v1-preregistered-20261010"
V1_GIT_TAG_OBJECT = "9982af3245ba94de22ad1e161f5f3003e1345c72"

SCRIPT_PATH = (
    PROJECT_ROOT / "benchmarks"
    / "jx_tno_stage15_resolved11_source_convergence_diagnostic_v2.py"
)
TEST_PATH = (
    PROJECT_ROOT / "tests"
    / "test_tno_stage15_resolved11_source_convergence_diagnostic_v2.py"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs"
    / "tno_stage15_resolved11_source_convergence_diagnostic_v2"
)


def _sha256_local(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


_v1_payload = V1_SOURCE_PATH.read_bytes()
if len(_v1_payload) != V1_SOURCE_BYTES or _sha256_local(_v1_payload) != V1_SOURCE_SHA256:
    raise RuntimeError("frozen source-convergence V1 implementation differs")
_BASE_NAMESPACE: dict[str, Any] = {
    "__name__": "_jx_tno_source_convergence_v1_bound_for_v2",
    "__file__": str(V1_SOURCE_PATH),
    "__package__": "",
}
exec(
    compile(
        _v1_payload, str(V1_SOURCE_PATH), "exec",
        optimize=sys.flags.optimize,
    ),
    _BASE_NAMESPACE,
)

_V1_BUILD_PROTOCOL = _BASE_NAMESPACE["build_protocol"]
_V1_INPUT_BINDINGS = _BASE_NAMESPACE["_input_bindings"]
_V1_VALIDATE_SMALL_PREREQUISITES = _BASE_NAMESPACE["_validate_small_prerequisites"]
_V1_PREPARE = _BASE_NAMESPACE["prepare"]
_V1_VALIDATE_PREPARED = _BASE_NAMESPACE["validate_prepared"]
_V1_VALIDATE_OUTPUT = _BASE_NAMESPACE["validate_output"]
_V1_EXECUTE = _BASE_NAMESPACE["execute"]

_LOCAL_SCHEMA_VALUES = {
    "PROTOCOL_SCHEMA": "jx.tno.stage15-resolved11-source-convergence-diagnostic.protocol.v2",
    "RESULT_SCHEMA": "jx.tno.stage15-resolved11-source-convergence-diagnostic.result.v2",
    "RUNTIME_SCHEMA": "jx.tno.stage15-resolved11-source-convergence-diagnostic.runtime.v2",
    "SOURCE_SCHEMA": "jx.tno.stage15-resolved11-source-convergence-diagnostic.source-nodes.v2",
    "ANNUAL_SCHEMA": "jx.tno.stage15-resolved11-source-convergence-diagnostic.annual.v2",
    "VALIDATION_SCHEMA": "jx.tno.stage15-resolved11-source-convergence-diagnostic.validation.v2",
    "PROTOCOL_DOMAIN": b"jx.tno.stage15-resolved11-source-convergence-diagnostic.protocol.v2\0",
    "RESULT_DOMAIN": b"jx.tno.stage15-resolved11-source-convergence-diagnostic.result.v2\0",
    "RUNTIME_DOMAIN": b"jx.tno.stage15-resolved11-source-convergence-diagnostic.runtime.v2\0",
    "VALIDATION_DOMAIN": b"jx.tno.stage15-resolved11-source-convergence-diagnostic.validation.v2\0",
    "SCRIPT_PATH": SCRIPT_PATH,
    "TEST_PATH": TEST_PATH,
    "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
}
_BASE_NAMESPACE.update(_LOCAL_SCHEMA_VALUES)


def _v1_file_binding(
    path: Path, size: int, digest: str, label: str,
) -> tuple[dict[str, Any], bytes]:
    payload = _BASE_NAMESPACE["_regular_bytes"](
        path, label, exact_bytes=size, maximum=max(size, 1),
    )
    if _sha256_local(payload) != digest:
        raise _BASE_NAMESPACE["SourceConvergenceDiagnosticError"](
            f"{label} digest differs"
        )
    return (
        {
            "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": size,
            "file_sha256": digest,
        },
        payload,
    )


def _validate_v1_predecessor() -> dict[str, Any]:
    source, _source_payload = _v1_file_binding(
        V1_SOURCE_PATH, V1_SOURCE_BYTES, V1_SOURCE_SHA256, "V1 implementation",
    )
    test, _test_payload = _v1_file_binding(
        V1_TEST_PATH, V1_TEST_BYTES, V1_TEST_SHA256, "V1 test",
    )
    protocol_binding, protocol_payload = _v1_file_binding(
        V1_PROTOCOL_PATH, V1_PROTOCOL_BYTES, V1_PROTOCOL_SHA256, "V1 protocol",
    )
    if (
        not V1_ROOT.is_dir() or V1_ROOT.is_symlink()
        or sorted(item.name for item in V1_ROOT.iterdir()) != ["PROTOCOL.json"]
    ):
        raise _BASE_NAMESPACE["SourceConvergenceDiagnosticError"](
            "V1 failed-attempt root is not exactly protocol-only"
    )
    protocol = _BASE_NAMESPACE["_load_canonical_json"](
        protocol_payload, "V1 protocol",
    )
    if (
        protocol.get("schema") != V1_PROTOCOL_SCHEMA
        or protocol.get("content_sha256") != V1_PROTOCOL_CONTENT_SHA256
        or _BASE_NAMESPACE["_domain_sha"](V1_PROTOCOL_DOMAIN, protocol)
        != V1_PROTOCOL_CONTENT_SHA256
        or protocol.get("implementation") != {**source, "test": test}
    ):
        raise _BASE_NAMESPACE["SourceConvergenceDiagnosticError"](
            "V1 predecessor protocol semantics differ"
        )
    protocol_binding["content_sha256"] = V1_PROTOCOL_CONTENT_SHA256
    return {"source": source, "test": test, "protocol": protocol_binding}


def _v2_input_bindings() -> list[dict[str, Any]]:
    bindings = list(_V1_INPUT_BINDINGS())
    predecessor = _validate_v1_predecessor()
    bindings.extend(
        predecessor[name] for name in ("source", "test", "protocol")
    )
    return bindings


def _v2_build_protocol() -> dict[str, Any]:
    predecessor = _validate_v1_predecessor()
    protocol = _V1_BUILD_PROTOCOL()
    protocol["status"] = "V2_PREDECLARED_ONE_SHOT_DIAGNOSTIC_NOT_EXECUTED"
    protocol["predecessor_v1"] = {
        "files": predecessor,
        "git_custody": {
            "branch": V1_GIT_BRANCH,
            "preregistration_commit": V1_GIT_COMMIT,
            "annotated_tag": V1_GIT_TAG,
            "tag_object": V1_GIT_TAG_OBJECT,
            "tag_peeled_commit": V1_GIT_COMMIT,
            "external_git_custody_not_reconstructed_by_this_program": True,
        },
        "failed_attempt": {
            "attempt_count": 1,
            "failure_phase": "POST_INTEGRATION_VALIDATION_BEFORE_PUBLICATION",
            "failure_id": "UNEQUAL_LENGTH_STRICT_ZIP_IN_IAS15_ANNUAL_STEP_MONOTONICITY_VALIDATOR",
            "failure_class": "ValueError",
            "failure_message": "zip() argument 2 is shorter than argument 1",
            "failing_expression": "STRICT_ZIP_OF_COMPLETE_ANNUAL_ROSTER_AND_ONE_ELEMENT_SHORT_TAIL",
            "mandatory_jx_integrations_completed": 24,
            "conditional_h1024_integrations_completed": 11,
            "tight_ias15_replicates_completed": 12,
            "all_integrations_completed": True,
            "source_only_dynamics_executed": True,
            "published_result_artifact_count": 0,
            "result_artifacts_published": False,
            "qualification_established": False,
            "classification_generated": False,
            "science_result": False,
            "retained_root_roster": ["PROTOCOL.json"],
        },
    }
    protocol["v2_fix"] = {
        "scope": "IAS15_ANNUAL_STEP_ADJACENCY_VALIDATION_ONLY",
        "annual_checkpoint_count": _BASE_NAMESPACE["CHECKPOINT_COUNT"],
        "adjacent_pair_count": _BASE_NAMESPACE["CHECKPOINT_COUNT"] - 1,
        "policy": "FOR_I_IN_0_THROUGH_999_REQUIRE_ANNUAL_I_PLUS_1_STRICTLY_GREATER_THAN_ANNUAL_I",
        "inherited_v1_adjacent_unequal_length_strict_zip_is_not_called": True,
        "all_other_physics_integrators_gates_dispositions_and_claims_unchanged": True,
    }
    protocol["content_sha256"] = _BASE_NAMESPACE["_domain_sha"](
        _BASE_NAMESPACE["PROTOCOL_DOMAIN"], protocol,
    )
    return protocol


def _v2_validate_small_prerequisites() -> tuple[Any, dict[str, Any]]:
    _validate_v1_predecessor()
    return _V1_VALIDATE_SMALL_PREREQUISITES()


def _v2_prepare(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _V1_PREPARE(root)


def _v2_validate_prepared(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _V1_VALIDATE_PREPARED(root)


def _v2_validate_output(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _V1_VALIDATE_OUTPUT(root)


def _v2_execute(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    return _V1_EXECUTE(root)


def _v2_validate_ias_receipt(value: Any) -> None:
    keys = {
        "epsilon_hex", "initial_dt_hex", "min_dt_hex", "adaptive_mode",
        "annual_integrate_call_count", "initial_steps_done", "final_steps_done",
        "annual_steps_done", "annual_steps_done_sha256", "final_time_hex",
        "gravitational_constant_hex", "gravity", "collision", "boundary",
        "softening_hex", "testparticle_type", "active_source_count",
        "source_count", "tracer_count",
    }
    _BASE_NAMESPACE["_require_keys"](value, keys, "tight IAS15 receipt")
    for key, minimum in (
        ("annual_integrate_call_count", 1), ("initial_steps_done", 0),
        ("final_steps_done", 1), ("testparticle_type", 0),
        ("active_source_count", 1), ("source_count", 1),
        ("tracer_count", 0),
    ):
        _BASE_NAMESPACE["_require_exact_int"](
            value[key], f"tight IAS15 {key}", minimum=minimum,
        )
    if (
        value["epsilon_hex"] != _BASE_NAMESPACE["IAS15_TIGHT_EPSILON"].hex()
        or value["initial_dt_hex"] != _BASE_NAMESPACE["IAS15_INITIAL_DT"].hex()
        or value["min_dt_hex"] != _BASE_NAMESPACE["IAS15_MIN_DT"].hex()
        or value["adaptive_mode"] != _BASE_NAMESPACE["IAS15_ADAPTIVE_MODE"]
        or value["annual_integrate_call_count"] != _BASE_NAMESPACE["HORIZON_YEARS"]
        or value["final_time_hex"] != float(_BASE_NAMESPACE["HORIZON_YEARS"]).hex()
        or value["gravitational_constant_hex"] != 1.0.hex()
        or value["gravity"] != "basic" or value["collision"] != "none"
        or value["boundary"] != "none" or value["softening_hex"] != 0.0.hex()
        or value["testparticle_type"] != 0
        or value["active_source_count"] != _BASE_NAMESPACE["MASSIVE_COUNT"]
        or value["source_count"] != _BASE_NAMESPACE["MASSIVE_COUNT"]
        or value["tracer_count"] != 0
    ):
        raise _BASE_NAMESPACE["SourceConvergenceDiagnosticError"](
            "tight IAS15 receipt semantics differ"
        )
    initial = _BASE_NAMESPACE["_require_exact_int"](
        value["initial_steps_done"], "IAS initial steps", minimum=0,
    )
    final = _BASE_NAMESPACE["_require_exact_int"](
        value["final_steps_done"], "IAS final steps", minimum=1,
    )
    annual = value["annual_steps_done"]
    checkpoint_count = _BASE_NAMESPACE["CHECKPOINT_COUNT"]
    if type(annual) is not list or len(annual) != checkpoint_count:
        raise _BASE_NAMESPACE["SourceConvergenceDiagnosticError"](
            "tight IAS15 annual step roster differs"
        )
    for index, item in enumerate(annual):
        _BASE_NAMESPACE["_require_exact_int"](
            item, f"tight IAS15 annual step {index}", minimum=0,
        )
    nonmonotonic = any(
        annual[index + 1] <= annual[index]
        for index in range(checkpoint_count - 1)
    )
    if (
        annual[0] != initial or annual[-1] != final or final <= initial
        or nonmonotonic
        or _sha256_local(struct.pack(f"<{len(annual)}Q", *annual))
        != value["annual_steps_done_sha256"]
    ):
        raise _BASE_NAMESPACE["SourceConvergenceDiagnosticError"](
            "tight IAS15 final step count differs"
        )
    _BASE_NAMESPACE["_require_sha"](
        value["annual_steps_done_sha256"], "IAS annual step digest",
    )


_BASE_NAMESPACE.update({
    "_input_bindings": _v2_input_bindings,
    "build_protocol": _v2_build_protocol,
    "_validate_small_prerequisites": _v2_validate_small_prerequisites,
    "_validate_ias_receipt": _v2_validate_ias_receipt,
    "prepare": _v2_prepare,
    "validate_prepared": _v2_validate_prepared,
    "validate_output": _v2_validate_output,
    "execute": _v2_execute,
})

for _name, _value in tuple(_BASE_NAMESPACE.items()):
    if _name not in {
        "__name__", "__file__", "__package__", "__builtins__",
        "__doc__", "__loader__", "__spec__",
    }:
        globals()[_name] = _value


if __name__ == "__main__":
    raise SystemExit(main())
