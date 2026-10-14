"""Path-free execution and verification for one portable JX engine job.

Engine Job V1 deliberately supports one narrow vertical slice: canonical
portable scenario-manifest bytes, NumPy/CPU adaptive RKF78 execution, one
deterministic trajectory artifact, one deterministic restart archive, and a
canonical receipt that binds their exact byte identities.  The receipt is an
integrity record, not a signature and not scientific qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import re
from typing import Any
import zipfile

from ._zip_preflight import preflight_exact_zip_roster
from .backends import BackendArrayError, BackendError, BackendUnavailableError
from .contracts import ContractError
from .evaluator import EvaluationError
from .force_abi import get_force_abi
from .forces import ForceError
from .integrator_registry import get_integrator_route
from .run_artifact import (
    ENGINE_RUN_ARTIFACT_HASH_ALGORITHM,
    ENGINE_RUN_ARTIFACT_MAX_BYTES,
    ENGINE_RUN_ARTIFACT_SCHEMA,
    ENGINE_RUN_ARTIFACT_SCOPE,
    EngineRunArtifactError,
    JXEngineRunArtifact,
    dump_engine_run_artifact,
    engine_run_artifact_sha256,
    load_engine_run_artifact,
)
from .scenario import DynamicsScenario, SCENARIO_EVIDENCE_CLASS
from .scenario_manifest import (
    PORTABLE_SCENARIO_HASH_ALGORITHM,
    PORTABLE_SCENARIO_MANIFEST_SCHEMA,
    PORTABLE_SCENARIO_MANIFEST_SCOPE,
    ScenarioManifestError,
    dynamics_scenario_manifest_sha256,
    load_dynamics_scenario_manifest,
)
from .simulation import (
    CONTINUATION_DIGEST_ALGORITHM,
    CONTINUATION_DIGEST_DOMAIN,
    SCIENTIFIC_CLAIM_STATE,
    JXSimulation,
    JXSimulationContractError,
    JXSimulationError,
    JXSimulationRun,
)
from .simulation_archive import (
    FORCE_PLAN_DIGEST_ALGORITHM,
    FORCE_PLAN_DIGEST_DOMAIN,
    SIMULATION_ARCHIVE_HASH_ALGORITHM,
    SIMULATION_ARCHIVE_MAX_BYTES,
    SIMULATION_ARCHIVE_SCHEMA,
    SIMULATION_ARCHIVE_SCOPE,
    JXSimulationArchiveLoad,
    SimulationArchiveError,
    dump_simulation_archive,
    load_simulation_archive,
    simulation_archive_sha256,
    simulation_force_plan_sha256,
)
from .trajectory import (
    RKF78_RESULT_CONTENT_CHECKSUM_ALGORITHM,
    RKF78_RESULT_CONTENT_CHECKSUM_DOMAIN,
    TrajectoryError,
)
from .trajectory_contracts import ADAPTIVE_RKF78_METHOD_ID, RKF78_STAGE_COUNT


ENGINE_RUN_RECEIPT_SCHEMA = "jxplanetx.engine-run-receipt.v1"
ENGINE_RUN_RECEIPT_SCOPE = "PORTABLE_NUMPY_CPU_RKF78_ENGINE_JOB_V1"
ENGINE_RUN_RECEIPT_HASH_ALGORITHM = "SHA256_EXACT_BYTES"
ENGINE_RUN_ID_DOMAIN = "jxplanetx.engine-run-id.v1"
ENGINE_RUN_ID_HASH_ALGORITHM = "SHA256_CANONICAL_ASCII_JSON_LF_SORTED_KEYS_V1"
ENGINE_RUN_RECEIPT_MAX_BYTES = 64 * 1024
ENGINE_JOB_BUNDLE_SCHEMA = "jxplanetx.engine-job-bundle.v1"
ENGINE_JOB_BUNDLE_HASH_ALGORITHM = "SHA256_EXACT_BUNDLE_BYTES"
ENGINE_JOB_EXPECTED_SHA256_SOURCE = (
    "CALLER_SUPPLIED_OUT_OF_BAND_EXACT_BYTES_NOT_AUTHENTICATION"
)
ENGINE_JOB_BUNDLE_MAX_BYTES = 128 * 1024 * 1024
ENGINE_JOB_V1_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
ENGINE_JOB_V1_MAX_RESTART_BYTES = 16 * 1024 * 1024
ENGINE_JOB_V1_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
ENGINE_JOB_V1_EXECUTION_POLICY = "BOUNDED_PORTABLE_NUMPY_CPU_RKF78_V1"
ENGINE_JOB_V1_MAX_BODIES = 256
ENGINE_JOB_V1_MAX_CHECKPOINTS = 4096
ENGINE_JOB_V1_MAX_FORCE_MODELS = 7
ENGINE_JOB_V1_MAXIMUM_STEPS = 100_000
ENGINE_JOB_V1_MAXIMUM_REJECTIONS = 10_000
ENGINE_JOB_V1_MAX_STAGE_BODY_PAIR_MODEL_EVALUATIONS = 500_000_000
ENGINE_JOB_V1_MAX_TEXT_ENCODING_BYTES = 4096
ENGINE_JOB_V1_MAX_IDENTITY_ENCODING_BYTES = 256 * 1024
_JOB_MEMBER_NAMES = (
    "trajectory.zip",
    "restart.zip",
    "receipt.json",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EngineRunnerError(JXSimulationContractError):
    """An Engine Job V1 input, execution, or bundle failed closed."""


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise EngineRunnerError(f"{label} must be lowercase SHA-256 hex")
    return value


def _canonical(document: object) -> bytes:
    try:
        rendered = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise EngineRunnerError("engine-run receipt is not canonical JSON") from exc
    raw = (rendered + "\n").encode("ascii")
    if len(raw) > ENGINE_RUN_RECEIPT_MAX_BYTES:
        raise EngineRunnerError("engine-run receipt exceeds the V1 byte ceiling")
    return raw


def _strict_json(raw: bytes) -> dict[str, Any]:
    duplicate: str | None = None

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, value in items:
            if key in result and duplicate is None:
                duplicate = key
            result[key] = value
        return result

    def reject_float(token: str) -> object:
        raise EngineRunnerError(
            f"JSON floating token {token!r} is forbidden in an engine-run receipt"
        )

    def reject_constant(token: str) -> object:
        raise EngineRunnerError(
            f"nonfinite JSON constant {token!r} is forbidden in an engine-run receipt"
        )

    if type(raw) is not bytes or not raw:
        raise EngineRunnerError("receipt_bytes must be nonempty exact bytes")
    if len(raw) > ENGINE_RUN_RECEIPT_MAX_BYTES:
        raise EngineRunnerError("engine-run receipt exceeds the V1 byte ceiling")
    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_float=reject_float,
            parse_constant=reject_constant,
        )
    except EngineRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise EngineRunnerError("engine-run receipt must be canonical ASCII JSON") from exc
    if duplicate is not None:
        raise EngineRunnerError(f"engine-run receipt repeats key {duplicate!r}")
    if type(document) is not dict or _canonical(document) != raw:
        raise EngineRunnerError("engine-run receipt bytes are not canonical")
    return document


def engine_run_receipt_sha256(receipt_bytes: bytes) -> str:
    """Return the exact canonical-receipt SHA-256 identity."""

    if type(receipt_bytes) is not bytes:
        raise EngineRunnerError("receipt_bytes must be exact bytes")
    return hashlib.sha256(receipt_bytes).hexdigest()


def engine_job_sha256(job_bytes: bytes) -> str:
    """Return the exact outer Engine Job V1 bundle identity."""

    if type(job_bytes) is not bytes:
        raise EngineRunnerError("job_bytes must be exact bytes")
    return hashlib.sha256(job_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class _RunProjection:
    simulation_id: str
    run_index: int
    integrator_id: str
    force_model_ids: tuple[str, ...]
    checkpoint_epochs: tuple[float, ...]
    checkpoint_positions: object
    checkpoint_velocities: object
    backend_id: str
    device: str
    native_result_digest: str
    native_result_digest_scope: str


def _projection_from_live_run(run: JXSimulationRun) -> _RunProjection:
    if type(run) is not JXSimulationRun or run.native_result_digest is None:
        raise EngineRunnerError(
            "Engine Job V1 requires an exact run with a retained content digest"
        )
    return _RunProjection(
        simulation_id=run.simulation_id,
        run_index=run.run_index,
        integrator_id=run.integrator_id,
        force_model_ids=run.force_model_ids,
        checkpoint_epochs=run.checkpoint_epochs,
        checkpoint_positions=run.checkpoint_positions,
        checkpoint_velocities=run.checkpoint_velocities,
        backend_id=run.backend_id,
        device=run.device,
        native_result_digest=run.native_result_digest,
        native_result_digest_scope=run.native_result_digest_scope,
    )


def _projection_from_artifact(artifact: JXEngineRunArtifact) -> _RunProjection:
    if type(artifact) is not JXEngineRunArtifact:
        raise EngineRunnerError("artifact must be an exact JXEngineRunArtifact")
    return _RunProjection(
        simulation_id=artifact.simulation_id,
        run_index=artifact.run_index,
        integrator_id=artifact.integrator_id,
        force_model_ids=artifact.force_model_ids,
        checkpoint_epochs=artifact.checkpoint_epochs,
        checkpoint_positions=artifact.checkpoint_positions,
        checkpoint_velocities=artifact.checkpoint_velocities,
        backend_id=artifact.backend_id,
        device=artifact.device,
        native_result_digest=artifact.native_result_digest,
        native_result_digest_scope=artifact.native_result_digest_scope,
    )


def _exact_array_equal(left: object, right: object) -> bool:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise EngineRunnerError("Engine Job V1 verification requires NumPy") from exc
    if type(left) is not np.ndarray or type(right) is not np.ndarray:
        return False
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.flags.c_contiguous
        and right.flags.c_contiguous
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _projections_equal(left: _RunProjection, right: _RunProjection) -> bool:
    return bool(
        left.simulation_id == right.simulation_id
        and left.run_index == right.run_index
        and left.integrator_id == right.integrator_id
        and left.force_model_ids == right.force_model_ids
        and tuple(value.hex() for value in left.checkpoint_epochs)
        == tuple(value.hex() for value in right.checkpoint_epochs)
        and _exact_array_equal(
            _stack_checkpoints(left.checkpoint_positions),
            _stack_checkpoints(right.checkpoint_positions),
        )
        and _exact_array_equal(
            _stack_checkpoints(left.checkpoint_velocities),
            _stack_checkpoints(right.checkpoint_velocities),
        )
        and left.backend_id == right.backend_id
        and left.device == right.device
        and left.native_result_digest == right.native_result_digest
        and left.native_result_digest_scope == right.native_result_digest_scope
    )


def _stack_checkpoints(value: object) -> object:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
        raise EngineRunnerError("Engine Job V1 verification requires NumPy") from exc
    if type(value) is np.ndarray:
        return value
    if type(value) is not tuple:
        return value
    try:
        result = np.stack(value, axis=0)
    except (TypeError, ValueError, MemoryError) as exc:
        raise EngineRunnerError("run checkpoint arrays could not be compared") from exc
    result = np.ascontiguousarray(result)
    return result


def _artifacts_equal(left: JXEngineRunArtifact, right: JXEngineRunArtifact) -> bool:
    excluded = {
        "checkpoint_epochs",
        "checkpoint_positions",
        "checkpoint_velocities",
        "_load_token",
    }
    scalar_equal = all(
        getattr(left, name) == getattr(right, name)
        for name in JXEngineRunArtifact.__dataclass_fields__
        if name not in excluded
    )
    return bool(
        scalar_equal
        and tuple(value.hex() for value in left.checkpoint_epochs)
        == tuple(value.hex() for value in right.checkpoint_epochs)
        and _exact_array_equal(
            left.checkpoint_positions,
            right.checkpoint_positions,
        )
        and _exact_array_equal(
            left.checkpoint_velocities,
            right.checkpoint_velocities,
        )
    )


def _snapshots_equal(left: object, right: object) -> bool:
    from .contracts import StateSnapshot

    if type(left) is not StateSnapshot or type(right) is not StateSnapshot:
        return False
    arrays = {
        "positions",
        "velocities",
        "gravitational_parameters",
        "masses",
        "radii",
        "massive",
    }
    scalar_equal = all(
        (
            getattr(left, name).hex() == getattr(right, name).hex()
            if name == "epoch"
            else getattr(left, name) == getattr(right, name)
        )
        for name in StateSnapshot.__dataclass_fields__
        if name not in arrays
    )
    return bool(
        scalar_equal
        and all(
            _exact_array_equal(getattr(left, name), getattr(right, name))
            for name in arrays
        )
    )


def _archives_equal(
    left: JXSimulationArchiveLoad,
    right: JXSimulationArchiveLoad,
) -> bool:
    left_continuation = left.continuation
    right_continuation = right.continuation
    return bool(
        left.archive_sha256 == right.archive_sha256
        and left.force_plan_sha256 == right.force_plan_sha256
        and left.schema == right.schema
        and left.scope == right.scope
        and left.scientific_claim_state == right.scientific_claim_state
        and left.production_authorized == right.production_authorized
        and all(
            getattr(left_continuation, name) == getattr(right_continuation, name)
            for name in (
                "simulation_id",
                "parent_run_index",
                "parent_integrator_id",
                "parent_native_result_digest",
                "continuation_state_sha256",
                "force_model_ids",
                "digest_algorithm",
                "digest_domain",
                "scientific_claim_state",
                "production_authorized",
            )
        )
        and _snapshots_equal(
            left_continuation.snapshot,
            right_continuation.snapshot,
        )
    )


def _archive_matches_projection(
    archive: JXSimulationArchiveLoad,
    projection: _RunProjection,
) -> bool:
    continuation = archive.continuation
    endpoint = continuation.snapshot
    positions = _stack_checkpoints(projection.checkpoint_positions)
    velocities = _stack_checkpoints(projection.checkpoint_velocities)
    return bool(
        continuation.simulation_id == projection.simulation_id
        and continuation.parent_run_index == projection.run_index
        and continuation.parent_integrator_id == projection.integrator_id
        and continuation.parent_native_result_digest
        == projection.native_result_digest
        and continuation.force_model_ids == projection.force_model_ids
        and endpoint.epoch.hex() == projection.checkpoint_epochs[-1].hex()
        and _exact_array_equal(endpoint.positions, positions[-1])
        and _exact_array_equal(endpoint.velocities, velocities[-1])
    )


def _validate_manifest_crossbindings(
    scenario: DynamicsScenario,
    manifest_sha256: str,
    artifact: JXEngineRunArtifact,
    archive: JXSimulationArchiveLoad,
    run_id: str,
) -> None:
    """Bind trusted manifest state to every retained trajectory/restart field."""

    expected_simulation_id = f"jx-engine-job-v1:{manifest_sha256}"
    initial = scenario.initial_snapshot
    projection = _projection_from_artifact(artifact)
    exact_coordinates = {
        "time_scale": initial.time_scale,
        "frame": initial.frame,
        "origin": initial.origin,
        "axes": initial.axes,
        "length_unit": initial.length_unit,
        "time_unit": initial.time_unit,
        "mass_unit": initial.mass_unit,
        "unit_system_id": initial.unit_system_id,
    }
    if (
        artifact.simulation_id != expected_simulation_id
        or artifact.run_id != run_id
        or artifact.run_index != 0
        or artifact.integrator_id != scenario.integration_spec.method_id
        or artifact.force_model_ids != scenario.force_model_ids
        or artifact.snapshot_id != initial.snapshot_id
        or artifact.plan_id != scenario.force_plan.plan_id
        or artifact.body_ids != initial.body_ids
        or artifact.backend_id != scenario.force_plan.backend.backend_id
        or artifact.device != scenario.force_plan.backend.device
        or any(
            getattr(artifact, field) != expected
            for field, expected in exact_coordinates.items()
        )
        or tuple(value.hex() for value in artifact.checkpoint_epochs)
        != tuple(
            float(value).hex()
            for value in scenario.integration_spec.checkpoint_epochs
        )
    ):
        raise EngineRunnerError(
            "trajectory artifact does not exactly represent the trusted manifest"
        )
    if (
        artifact.accepted_steps < artifact.checkpoint_count - 1
        or artifact.attempted_steps > scenario.integration_spec.maximum_steps
        or artifact.rejected_steps > scenario.integration_spec.maximum_rejections
        or artifact.force_evaluations != artifact.attempted_steps * RKF78_STAGE_COUNT
    ):
        raise EngineRunnerError("trajectory artifact controller accounting is impossible")
    if not (
        _exact_array_equal(artifact.checkpoint_positions[0], initial.positions)
        and _exact_array_equal(artifact.checkpoint_velocities[0], initial.velocities)
    ):
        raise EngineRunnerError(
            "trajectory artifact initial state differs bitwise from the manifest"
        )

    continuation = archive.continuation
    restarted = continuation.snapshot
    expected_state_sha256 = continuation.continuation_state_sha256
    expected_snapshot_id = (
        f"{initial.snapshot_id}.continuation.0.{expected_state_sha256[:16]}"
    )
    expected_provenance = {
        "source_id": f"jx-continuation:{expected_simulation_id}:0",
        "citation": f"JX continuation of run 0 ({projection.integrator_id})",
        "version": "1",
        "sha256": expected_state_sha256,
    }
    if (
        not _archive_matches_projection(archive, projection)
        or restarted.body_ids != initial.body_ids
        or restarted.snapshot_id != expected_snapshot_id
        or any(
            getattr(restarted.provenance, field) != expected
            for field, expected in expected_provenance.items()
        )
        or any(
            getattr(restarted, field) != expected
            for field, expected in exact_coordinates.items()
        )
    ):
        raise EngineRunnerError(
            "restart archive is not the trajectory artifact endpoint"
        )
    if not (
        _exact_array_equal(restarted.gravitational_parameters, initial.gravitational_parameters)
        and _exact_array_equal(restarted.masses, initial.masses)
        and _exact_array_equal(restarted.radii, initial.radii)
        and _exact_array_equal(restarted.massive, initial.massive)
    ):
        raise EngineRunnerError(
            "restart archive retained state constants differ from the manifest"
        )


def _run_id(
    *,
    manifest_sha256: str,
    force_plan_sha256: str,
    run: _RunProjection,
) -> str:
    seed = _canonical(
        {
            "domain": ENGINE_RUN_ID_DOMAIN,
            "force_plan_sha256": force_plan_sha256,
            "integrator_id": run.integrator_id,
            "manifest_sha256": manifest_sha256,
            "native_result_digest": run.native_result_digest,
            "run_index": run.run_index,
            "simulation_id": run.simulation_id,
        }
    )
    return hashlib.sha256(seed).hexdigest()


def engine_job_claim_ceiling() -> dict[str, object]:
    """Return a fresh copy of the fixed nonauthorizing Engine Job V1 claim block."""

    return {
        "accuracy_claimed": False,
        "content_integrity_only": True,
        "cross_platform_replay_claimed": False,
        "deterministic_encoding_only": True,
        "evidence_class": SCENARIO_EVIDENCE_CLASS,
        "execution_authenticated": False,
        "general_superiority_claimed": False,
        "improvement_claimed": False,
        "native_result_digest_recomputed": False,
        "numerical_execution_replayed": False,
        "physics_claimed": False,
        "production_authorized": False,
        "qualification_authorized": False,
        "registry_authorized": False,
        "scientific_claim_state": SCIENTIFIC_CLAIM_STATE,
        "source_runtime_reproduction_claimed": False,
    }


def _receipt_document(
    *,
    manifest_sha256: str,
    manifest_size: int,
    scenario_id: str,
    force_plan_sha256: str,
    run_id: str,
    run: _RunProjection,
    artifact_sha256: str,
    artifact_size: int,
    archive_sha256: str,
    archive_size: int,
    continuation_state_sha256: str,
    parent_run_index: int,
    parent_integrator_id: str,
    parent_native_result_digest: str,
) -> dict[str, object]:
    for value, label in (
        (manifest_sha256, "manifest_sha256"),
        (force_plan_sha256, "force_plan_sha256"),
        (run_id, "run_id"),
        (artifact_sha256, "artifact_sha256"),
        (archive_sha256, "archive_sha256"),
        (continuation_state_sha256, "continuation_state_sha256"),
        (parent_native_result_digest, "parent_native_result_digest"),
    ):
        _digest(value, label)
    for value, label in (
        (manifest_size, "manifest_size"),
        (artifact_size, "artifact_size"),
        (archive_size, "archive_size"),
    ):
        if type(value) is not int or value <= 0:
            raise EngineRunnerError(f"{label} must be a positive exact integer")
    if type(scenario_id) is not str or not scenario_id or scenario_id.strip() != scenario_id:
        raise EngineRunnerError("scenario_id must be nonempty trimmed text")
    if type(parent_run_index) is not int or parent_run_index < 0:
        raise EngineRunnerError("parent_run_index must be a nonnegative exact integer")
    if (
        type(parent_integrator_id) is not str
        or not parent_integrator_id
        or parent_integrator_id.strip() != parent_integrator_id
    ):
        raise EngineRunnerError("parent_integrator_id must be nonempty trimmed text")
    if type(run) is not _RunProjection:
        raise EngineRunnerError("run must be an exact internal run projection")
    return {
        "bundle": {
            "expected_sha256_source": ENGINE_JOB_EXPECTED_SHA256_SOURCE,
            "hash_algorithm": ENGINE_JOB_BUNDLE_HASH_ALGORITHM,
            "schema": ENGINE_JOB_BUNDLE_SCHEMA,
        },
        "claim": engine_job_claim_ceiling(),
        "execution_policy": {
            "id": ENGINE_JOB_V1_EXECUTION_POLICY,
            "maximum_bodies": ENGINE_JOB_V1_MAX_BODIES,
            "maximum_checkpoints": ENGINE_JOB_V1_MAX_CHECKPOINTS,
            "maximum_force_models": ENGINE_JOB_V1_MAX_FORCE_MODELS,
            "maximum_rejections": ENGINE_JOB_V1_MAXIMUM_REJECTIONS,
            "maximum_stage_body_pair_model_evaluations": (
                ENGINE_JOB_V1_MAX_STAGE_BODY_PAIR_MODEL_EVALUATIONS
            ),
            "maximum_steps": ENGINE_JOB_V1_MAXIMUM_STEPS,
            "maximum_text_encoding_bytes": ENGINE_JOB_V1_MAX_TEXT_ENCODING_BYTES,
            "maximum_total_identity_encoding_bytes": (
                ENGINE_JOB_V1_MAX_IDENTITY_ENCODING_BYTES
            ),
        },
        "force_plan": {
            "digest_algorithm": FORCE_PLAN_DIGEST_ALGORITHM,
            "digest_domain": FORCE_PLAN_DIGEST_DOMAIN,
            "sha256": force_plan_sha256,
        },
        "hash_algorithm": ENGINE_RUN_RECEIPT_HASH_ALGORITHM,
        "manifest": {
            "hash_algorithm": PORTABLE_SCENARIO_HASH_ALGORITHM,
            "schema": PORTABLE_SCENARIO_MANIFEST_SCHEMA,
            "scenario_id": scenario_id,
            "sha256": manifest_sha256,
            "size_bytes": manifest_size,
            "scope": PORTABLE_SCENARIO_MANIFEST_SCOPE,
        },
        "restart_archive": {
            "continuation_digest_algorithm": CONTINUATION_DIGEST_ALGORITHM,
            "continuation_digest_domain": CONTINUATION_DIGEST_DOMAIN,
            "continuation_state_sha256": continuation_state_sha256,
            "parent_integrator_id": parent_integrator_id,
            "parent_native_result_digest": parent_native_result_digest,
            "parent_run_index": parent_run_index,
            "hash_algorithm": SIMULATION_ARCHIVE_HASH_ALGORITHM,
            "schema": SIMULATION_ARCHIVE_SCHEMA,
            "sha256": archive_sha256,
            "size_bytes": archive_size,
            "scope": SIMULATION_ARCHIVE_SCOPE,
        },
        "run": {
            "backend_id": run.backend_id,
            "device": run.device,
            "force_model_ids": list(run.force_model_ids),
            "integrator_id": run.integrator_id,
            "native_result_digest": run.native_result_digest,
            "native_result_digest_algorithm": (
                RKF78_RESULT_CONTENT_CHECKSUM_ALGORITHM
            ),
            "native_result_digest_domain": RKF78_RESULT_CONTENT_CHECKSUM_DOMAIN,
            "native_result_digest_scope": run.native_result_digest_scope,
            "native_result_digest_verification": (
                "DECLARED_CROSS_BOUND_NOT_RECOMPUTED"
            ),
            "run_index": run.run_index,
            "simulation_id": run.simulation_id,
        },
        "run_identity": {
            "algorithm": ENGINE_RUN_ID_HASH_ALGORITHM,
            "domain": ENGINE_RUN_ID_DOMAIN,
            "sha256": run_id,
        },
        "schema": ENGINE_RUN_RECEIPT_SCHEMA,
        "scope": ENGINE_RUN_RECEIPT_SCOPE,
        "trajectory_artifact": {
            "hash_algorithm": ENGINE_RUN_ARTIFACT_HASH_ALGORITHM,
            "schema": ENGINE_RUN_ARTIFACT_SCHEMA,
            "sha256": artifact_sha256,
            "size_bytes": artifact_size,
            "scope": ENGINE_RUN_ARTIFACT_SCOPE,
        },
    }


@dataclass(frozen=True, slots=True, eq=False)
class EngineRunBundle:
    """Exact in-memory output bytes from one successful Engine Job V1 run."""

    run_id: str
    manifest_sha256: str
    manifest_bytes: bytes
    artifact_bytes: bytes
    artifact_sha256: str
    restart_archive_bytes: bytes
    restart_archive_sha256: str
    receipt_bytes: bytes
    receipt_sha256: str
    job_bytes: bytes
    job_sha256: str
    run: JXSimulationRun
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    qualification_authorized: bool = False
    registry_authorized: bool = False
    execution_authenticated: bool = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_id, "run_id"),
            (self.manifest_sha256, "manifest_sha256"),
            (self.artifact_sha256, "artifact_sha256"),
            (self.restart_archive_sha256, "restart_archive_sha256"),
            (self.receipt_sha256, "receipt_sha256"),
            (self.job_sha256, "job_sha256"),
        ):
            _digest(value, label)
        for value, label in (
            (self.manifest_bytes, "manifest_bytes"),
            (self.artifact_bytes, "artifact_bytes"),
            (self.restart_archive_bytes, "restart_archive_bytes"),
            (self.receipt_bytes, "receipt_bytes"),
            (self.job_bytes, "job_bytes"),
        ):
            if type(value) is not bytes or not value:
                raise EngineRunnerError(f"{label} must be nonempty exact bytes")
        if type(self.run) is not JXSimulationRun:
            raise EngineRunnerError("run must be an exact JXSimulationRun")
        if (
            engine_run_artifact_sha256(self.artifact_bytes) != self.artifact_sha256
            or simulation_archive_sha256(self.restart_archive_bytes)
            != self.restart_archive_sha256
            or engine_run_receipt_sha256(self.receipt_bytes) != self.receipt_sha256
            or engine_job_sha256(self.job_bytes) != self.job_sha256
        ):
            raise EngineRunnerError("bundle byte identities are inconsistent")
        if _dump_job_bundle(
            self.artifact_bytes,
            self.restart_archive_bytes,
            self.receipt_bytes,
        ) != self.job_bytes:
            raise EngineRunnerError(
                "job_bytes are not the canonical bundle of the retained members"
            )
        try:
            _validate_engine_job_v1_manifest_bytes(self.manifest_bytes)
            scenario = load_dynamics_scenario_manifest(
                self.manifest_bytes,
                self.manifest_sha256,
            )
            _validate_engine_job_v1_scenario(scenario)
            artifact = load_engine_run_artifact(
                self.artifact_bytes,
                expected_sha256=self.artifact_sha256,
                expected_manifest_sha256=self.manifest_sha256,
                expected_run_id=self.run_id,
            )
            if (
                dump_engine_run_artifact(
                    self.run,
                    manifest_sha256=self.manifest_sha256,
                    run_id=self.run_id,
                )
                != self.artifact_bytes
            ):
                raise EngineRunnerError(
                    "bundle artifact is not the exact retained live-run projection"
                )
            live_projection = _projection_from_live_run(self.run)
            artifact_projection = _projection_from_artifact(artifact)
            native_force_plan = self.run.native_result.force_plan
            force_plan_sha256 = simulation_force_plan_sha256(scenario.force_plan)
            if simulation_force_plan_sha256(native_force_plan) != force_plan_sha256:
                raise EngineRunnerError(
                    "bundle live run force plan differs from the trusted manifest"
                )
            archive = load_simulation_archive(
                self.restart_archive_bytes,
                expected_sha256=self.restart_archive_sha256,
                force_plan=scenario.force_plan,
            )
            _strict_json(self.receipt_bytes)
            continuation = archive.continuation
            if (
                artifact.simulation_id
                != f"jx-engine-job-v1:{self.manifest_sha256}"
                or not _projections_equal(live_projection, artifact_projection)
                or _run_id(
                    manifest_sha256=self.manifest_sha256,
                    force_plan_sha256=force_plan_sha256,
                    run=artifact_projection,
                )
                != self.run_id
            ):
                raise EngineRunnerError(
                    "bundle identities do not match its retained live run"
                )
            _validate_manifest_crossbindings(
                scenario,
                self.manifest_sha256,
                artifact,
                archive,
                self.run_id,
            )
            expected_receipt = _receipt_document(
                manifest_sha256=self.manifest_sha256,
                manifest_size=len(self.manifest_bytes),
                scenario_id=scenario.scenario_id,
                force_plan_sha256=force_plan_sha256,
                run_id=self.run_id,
                run=artifact_projection,
                artifact_sha256=self.artifact_sha256,
                artifact_size=len(self.artifact_bytes),
                archive_sha256=self.restart_archive_sha256,
                archive_size=len(self.restart_archive_bytes),
                continuation_state_sha256=(
                    continuation.continuation_state_sha256
                ),
                parent_run_index=continuation.parent_run_index,
                parent_integrator_id=continuation.parent_integrator_id,
                parent_native_result_digest=(
                    continuation.parent_native_result_digest
                ),
            )
            if self.receipt_bytes != _canonical(expected_receipt):
                raise EngineRunnerError(
                    "bundle receipt is not its exact retained-member projection"
                )
        except EngineRunnerError:
            raise
        except (
            AttributeError,
            EngineRunArtifactError,
            KeyError,
            SimulationArchiveError,
            TypeError,
            ValueError,
        ) as exc:
            raise EngineRunnerError(
                "bundle semantic identities are inconsistent"
            ) from exc
        if (
            self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
            or type(self.qualification_authorized) is not bool
            or self.qualification_authorized
            or type(self.registry_authorized) is not bool
            or self.registry_authorized
            or type(self.execution_authenticated) is not bool
            or self.execution_authenticated
        ):
            raise EngineRunnerError("Engine Job V1 remains screening-only")


@dataclass(frozen=True, slots=True, eq=False)
class EngineRunVerification:
    """Content verification result for one complete Engine Job V1 bundle."""

    run_id: str
    manifest_sha256: str
    artifact_sha256: str
    restart_archive_sha256: str
    receipt_sha256: str
    job_sha256: str
    manifest_bytes: bytes
    job_bytes: bytes
    artifact: JXEngineRunArtifact
    archive: JXSimulationArchiveLoad
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    qualification_authorized: bool = False
    registry_authorized: bool = False
    execution_authenticated: bool = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_id, "run_id"),
            (self.manifest_sha256, "manifest_sha256"),
            (self.artifact_sha256, "artifact_sha256"),
            (self.restart_archive_sha256, "restart_archive_sha256"),
            (self.receipt_sha256, "receipt_sha256"),
            (self.job_sha256, "job_sha256"),
        ):
            _digest(value, label)
        if type(self.artifact) is not JXEngineRunArtifact:
            raise EngineRunnerError("artifact must be an exact JXEngineRunArtifact")
        if type(self.archive) is not JXSimulationArchiveLoad:
            raise EngineRunnerError("archive must be an exact JXSimulationArchiveLoad")
        if type(self.manifest_bytes) is not bytes or not self.manifest_bytes:
            raise EngineRunnerError("manifest_bytes must be nonempty exact bytes")
        if type(self.job_bytes) is not bytes or not self.job_bytes:
            raise EngineRunnerError("job_bytes must be nonempty exact bytes")
        try:
            _validate_engine_job_v1_manifest_bytes(self.manifest_bytes)
            scenario = load_dynamics_scenario_manifest(
                self.manifest_bytes,
                self.manifest_sha256,
            )
            _validate_engine_job_v1_scenario(scenario)
            artifact_bytes, archive_bytes, receipt_bytes = _load_job_bundle(
                self.job_bytes,
                self.job_sha256,
            )
            loaded_artifact = load_engine_run_artifact(
                artifact_bytes,
                expected_sha256=self.artifact_sha256,
                expected_manifest_sha256=self.manifest_sha256,
                expected_run_id=self.run_id,
            )
            loaded_archive = load_simulation_archive(
                archive_bytes,
                expected_sha256=self.restart_archive_sha256,
                force_plan=scenario.force_plan,
            )
            receipt = _strict_json(receipt_bytes)
            projection = _projection_from_artifact(loaded_artifact)
            continuation = loaded_archive.continuation
            expected_receipt = _receipt_document(
                manifest_sha256=self.manifest_sha256,
                manifest_size=len(self.manifest_bytes),
                scenario_id=scenario.scenario_id,
                force_plan_sha256=loaded_archive.force_plan_sha256,
                run_id=self.run_id,
                run=projection,
                artifact_sha256=self.artifact_sha256,
                artifact_size=len(artifact_bytes),
                archive_sha256=self.restart_archive_sha256,
                archive_size=len(archive_bytes),
                continuation_state_sha256=(
                    continuation.continuation_state_sha256
                ),
                parent_run_index=continuation.parent_run_index,
                parent_integrator_id=continuation.parent_integrator_id,
                parent_native_result_digest=(
                    continuation.parent_native_result_digest
                ),
            )
            if (
                engine_run_receipt_sha256(receipt_bytes) != self.receipt_sha256
                or self.artifact.artifact_sha256 != self.artifact_sha256
                or self.artifact.manifest_sha256 != self.manifest_sha256
                or self.artifact.run_id != self.run_id
                or self.archive.archive_sha256 != self.restart_archive_sha256
                or not _artifacts_equal(self.artifact, loaded_artifact)
                or not _archives_equal(self.archive, loaded_archive)
                or receipt["manifest"]["sha256"] != self.manifest_sha256
                or receipt["run_identity"]["sha256"] != self.run_id
                or receipt["trajectory_artifact"]["sha256"]
                != self.artifact_sha256
                or receipt["restart_archive"]["sha256"]
                != self.restart_archive_sha256
                or receipt_bytes != _canonical(expected_receipt)
                or _run_id(
                    manifest_sha256=self.manifest_sha256,
                    force_plan_sha256=loaded_archive.force_plan_sha256,
                    run=projection,
                )
                != self.run_id
                or not _archive_matches_projection(loaded_archive, projection)
            ):
                raise EngineRunnerError(
                    "verification envelope identities are inconsistent"
                )
            _validate_manifest_crossbindings(
                scenario,
                self.manifest_sha256,
                loaded_artifact,
                loaded_archive,
                self.run_id,
            )
        except EngineRunnerError:
            raise
        except (
            EngineRunArtifactError,
            KeyError,
            SimulationArchiveError,
            TypeError,
            ValueError,
        ) as exc:
            raise EngineRunnerError(
                "verification envelope could not revalidate its retained bytes"
            ) from exc
        if (
            self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
            or type(self.qualification_authorized) is not bool
            or self.qualification_authorized
            or type(self.registry_authorized) is not bool
            or self.registry_authorized
            or type(self.execution_authenticated) is not bool
            or self.execution_authenticated
        ):
            raise EngineRunnerError("verified Engine Job V1 data remains screening-only")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _dump_job_bundle(
    artifact_bytes: bytes,
    archive_bytes: bytes,
    receipt_bytes: bytes,
) -> bytes:
    members = {
        "trajectory.zip": artifact_bytes,
        "restart.zip": archive_bytes,
        "receipt.json": receipt_bytes,
    }
    if any(type(value) is not bytes or not value for value in members.values()):
        raise EngineRunnerError("job members must be nonempty exact bytes")
    if (
        len(artifact_bytes) > ENGINE_JOB_V1_MAX_ARTIFACT_BYTES
        or len(archive_bytes) > ENGINE_JOB_V1_MAX_RESTART_BYTES
        or len(receipt_bytes) > ENGINE_RUN_RECEIPT_MAX_BYTES
    ):
        raise EngineRunnerError("job member exceeds its Engine Job V1 byte ceiling")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in _JOB_MEMBER_NAMES:
            archive.writestr(_zip_info(name), members[name])
    result = stream.getvalue()
    if len(result) > ENGINE_JOB_BUNDLE_MAX_BYTES:
        raise EngineRunnerError("engine-job bundle exceeds the V1 byte ceiling")
    return result


def _load_job_bundle(job_bytes: bytes, expected_sha256: str) -> tuple[bytes, bytes, bytes]:
    if type(job_bytes) is not bytes or not job_bytes:
        raise EngineRunnerError("job_bytes must be nonempty exact bytes")
    if len(job_bytes) > ENGINE_JOB_BUNDLE_MAX_BYTES:
        raise EngineRunnerError("engine-job bundle exceeds the V1 byte ceiling")
    if engine_job_sha256(job_bytes) != _digest(expected_sha256, "expected_job_sha256"):
        raise EngineRunnerError("engine-job bundle SHA-256 identity mismatch")
    try:
        preflight_exact_zip_roster(job_bytes, _JOB_MEMBER_NAMES)
        context = zipfile.ZipFile(io.BytesIO(job_bytes), "r")
    except ValueError as exc:
        raise EngineRunnerError(
            "engine-job member roster or central directory is not canonical"
        ) from exc
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise EngineRunnerError("engine-job bytes are not a valid ZIP container") from exc
    with context as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        if names != _JOB_MEMBER_NAMES or len(set(names)) != len(names):
            raise EngineRunnerError("engine-job member roster or order is not canonical")
        total = 0
        member_limits = {
            "trajectory.zip": min(
                ENGINE_RUN_ARTIFACT_MAX_BYTES,
                ENGINE_JOB_V1_MAX_ARTIFACT_BYTES,
            ),
            "restart.zip": min(
                SIMULATION_ARCHIVE_MAX_BYTES,
                ENGINE_JOB_V1_MAX_RESTART_BYTES,
            ),
            "receipt.json": ENGINE_RUN_RECEIPT_MAX_BYTES,
        }
        for info in infos:
            total += info.file_size
            mode = (info.external_attr >> 16) & 0o170000
            if (
                info.is_dir()
                or info.compress_type != zipfile.ZIP_STORED
                or info.file_size <= 0
                or info.file_size > member_limits[info.filename]
                or info.compress_size != info.file_size
                or mode not in (0, 0o100000)
            ):
                raise EngineRunnerError("engine-job contains an invalid member")
        if total > ENGINE_JOB_BUNDLE_MAX_BYTES:
            raise EngineRunnerError("engine-job members exceed the V1 byte ceiling")
        try:
            artifact = archive.read("trajectory.zip")
            restart = archive.read("restart.zip")
            receipt = archive.read("receipt.json")
        except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise EngineRunnerError("engine-job members could not be read safely") from exc
    if _dump_job_bundle(artifact, restart, receipt) != job_bytes:
        raise EngineRunnerError("engine-job container bytes are not canonical")
    return artifact, restart, receipt


def _validate_engine_job_v1_scenario(scenario: DynamicsScenario) -> object:
    """Return the route after enforcing the complete V1 execution envelope."""

    if type(scenario) is not DynamicsScenario:
        raise EngineRunnerError("scenario must be an exact DynamicsScenario")
    route = get_integrator_route(scenario.integration_spec.method_id)
    backend = scenario.force_plan.backend
    if (
        route.integrator_id != ADAPTIVE_RKF78_METHOD_ID
        or backend.backend_id != "numpy"
        or backend.device != "cpu"
        or not route.supports_portable_manifest(
            backend.backend_id,
            backend.device,
            scenario.force_model_ids,
        )
    ):
        raise EngineRunnerError(
            "Engine Job V1 supports only portable NumPy/CPU adaptive RKF78"
        )
    for model in scenario.force_plan.models:
        force_abi = get_force_abi(model)
        if (
            force_abi.model_id != model.model_id
            or backend.backend_id not in force_abi.supported_backend_ids
            or "GENERAL_FIRST_ORDER" not in force_abi.compatible_integrator_classes
        ):
            raise EngineRunnerError(
                f"force model {model.model_id!r} is outside Engine Job V1"
            )

    snapshot = scenario.initial_snapshot
    identity_text = (
        scenario.scenario_id,
        snapshot.snapshot_id,
        scenario.force_plan.plan_id,
        *snapshot.body_ids,
        snapshot.time_scale,
        snapshot.frame,
        snapshot.origin,
        snapshot.axes,
        snapshot.length_unit,
        snapshot.time_unit,
        snapshot.mass_unit,
        snapshot.unit_system_id,
        *(model.model_id for model in scenario.force_plan.models),
    )
    try:
        encoded_sizes = tuple(
            len(json.dumps(value, ensure_ascii=True).encode("ascii"))
            for value in identity_text
        )
    except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise EngineRunnerError(
            "Engine Job V1 identity metadata is not canonically encodable"
        ) from exc
    if (
        any(size > ENGINE_JOB_V1_MAX_TEXT_ENCODING_BYTES for size in encoded_sizes)
        or sum(encoded_sizes) > ENGINE_JOB_V1_MAX_IDENTITY_ENCODING_BYTES
    ):
        raise EngineRunnerError(
            "manifest identity metadata exceeds the Engine Job V1 encoding policy"
        )

    body_count = len(scenario.initial_snapshot.body_ids)
    checkpoint_count = len(scenario.integration_spec.checkpoint_epochs)
    force_model_count = len(scenario.force_plan.models)
    requested_pair_model_work = (
        body_count
        * body_count
        * force_model_count
        * scenario.integration_spec.maximum_steps
        * RKF78_STAGE_COUNT
    )
    if (
        body_count > ENGINE_JOB_V1_MAX_BODIES
        or checkpoint_count > ENGINE_JOB_V1_MAX_CHECKPOINTS
        or force_model_count > ENGINE_JOB_V1_MAX_FORCE_MODELS
        or scenario.integration_spec.maximum_steps
        > ENGINE_JOB_V1_MAXIMUM_STEPS
        or scenario.integration_spec.maximum_rejections
        > ENGINE_JOB_V1_MAXIMUM_REJECTIONS
        or requested_pair_model_work
        > ENGINE_JOB_V1_MAX_STAGE_BODY_PAIR_MODEL_EVALUATIONS
    ):
        raise EngineRunnerError(
            "manifest exceeds the bounded Engine Job V1 execution policy"
        )
    return route


def _validate_engine_job_v1_manifest_bytes(manifest_bytes: bytes) -> None:
    if (
        type(manifest_bytes) is not bytes
        or not manifest_bytes
        or len(manifest_bytes) > ENGINE_JOB_V1_MAX_MANIFEST_BYTES
    ):
        raise EngineRunnerError(
            "manifest_bytes must be nonempty exact bytes within the Engine Job V1 ceiling"
        )


def run_scenario_manifest(
    manifest_bytes: bytes,
    expected_sha256: str,
) -> EngineRunBundle:
    """Execute one verified portable manifest entirely in memory."""

    try:
        _validate_engine_job_v1_manifest_bytes(manifest_bytes)
        scenario = load_dynamics_scenario_manifest(manifest_bytes, expected_sha256)
        manifest_sha256 = dynamics_scenario_manifest_sha256(manifest_bytes)
        route = _validate_engine_job_v1_scenario(scenario)

        simulation_id = f"jx-engine-job-v1:{manifest_sha256}"
        simulation = JXSimulation(
            simulation_id,
            scenario.initial_snapshot,
            scenario.force_plan,
        )
        run = simulation.integrate(route.integrator_id, scenario.integration_spec)
        force_plan_sha256 = simulation_force_plan_sha256(simulation.force_plan)
        projection = _projection_from_live_run(run)
        run_id = _run_id(
            manifest_sha256=manifest_sha256,
            force_plan_sha256=force_plan_sha256,
            run=projection,
        )
        artifact_bytes = dump_engine_run_artifact(
            run,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
        )
        artifact_sha256 = engine_run_artifact_sha256(artifact_bytes)
        continuation = simulation.prepare_continuation(run.run_index)
        restart_archive_bytes = dump_simulation_archive(simulation, run.run_index)
        restart_archive_sha256 = simulation_archive_sha256(restart_archive_bytes)
        receipt_bytes = _canonical(
            _receipt_document(
                manifest_sha256=manifest_sha256,
                manifest_size=len(manifest_bytes),
                scenario_id=scenario.scenario_id,
                force_plan_sha256=force_plan_sha256,
                run_id=run_id,
                run=projection,
                artifact_sha256=artifact_sha256,
                artifact_size=len(artifact_bytes),
                archive_sha256=restart_archive_sha256,
                archive_size=len(restart_archive_bytes),
                continuation_state_sha256=continuation.continuation_state_sha256,
                parent_run_index=continuation.parent_run_index,
                parent_integrator_id=continuation.parent_integrator_id,
                parent_native_result_digest=(
                    continuation.parent_native_result_digest
                ),
            )
        )
        job_bytes = _dump_job_bundle(
            artifact_bytes,
            restart_archive_bytes,
            receipt_bytes,
        )
        return EngineRunBundle(
            run_id=run_id,
            manifest_sha256=manifest_sha256,
            manifest_bytes=manifest_bytes,
            artifact_bytes=artifact_bytes,
            artifact_sha256=artifact_sha256,
            restart_archive_bytes=restart_archive_bytes,
            restart_archive_sha256=restart_archive_sha256,
            receipt_bytes=receipt_bytes,
            receipt_sha256=engine_run_receipt_sha256(receipt_bytes),
            job_bytes=job_bytes,
            job_sha256=engine_job_sha256(job_bytes),
            run=run,
        )
    except EngineRunnerError:
        raise
    except (
        EngineRunArtifactError,
        BackendArrayError,
        BackendError,
        BackendUnavailableError,
        ContractError,
        EvaluationError,
        ForceError,
        ScenarioManifestError,
        SimulationArchiveError,
        JXSimulationContractError,
        JXSimulationError,
        TrajectoryError,
        KeyError,
        RecursionError,
    ) as exc:
        raise EngineRunnerError(f"Engine Job V1 failed closed: {exc}") from exc


def _verify_engine_run_components(
    manifest_bytes: bytes,
    *,
    expected_manifest_sha256: str,
    artifact_bytes: bytes,
    restart_archive_bytes: bytes,
    receipt_bytes: bytes,
    expected_receipt_sha256: str,
    expected_job_sha256: str,
) -> EngineRunVerification:
    """Verify one bundle without re-running its numerical integration."""

    try:
        _validate_engine_job_v1_manifest_bytes(manifest_bytes)
        expected_manifest_sha256 = _digest(
            expected_manifest_sha256, "expected_manifest_sha256"
        )
        expected_receipt_sha256 = _digest(
            expected_receipt_sha256, "expected_receipt_sha256"
        )
        if engine_run_receipt_sha256(receipt_bytes) != expected_receipt_sha256:
            raise EngineRunnerError("receipt SHA-256 identity mismatch")
        scenario = load_dynamics_scenario_manifest(
            manifest_bytes, expected_manifest_sha256
        )
        _validate_engine_job_v1_scenario(scenario)
        receipt = _strict_json(receipt_bytes)
        if set(receipt) != {
            "bundle",
            "claim",
            "execution_policy",
            "force_plan",
            "hash_algorithm",
            "manifest",
            "restart_archive",
            "run",
            "run_identity",
            "schema",
            "scope",
            "trajectory_artifact",
        }:
            raise EngineRunnerError("engine-run receipt key roster changed")
        if (
            receipt["schema"] != ENGINE_RUN_RECEIPT_SCHEMA
            or receipt["scope"] != ENGINE_RUN_RECEIPT_SCOPE
            or receipt["hash_algorithm"] != ENGINE_RUN_RECEIPT_HASH_ALGORITHM
        ):
            raise EngineRunnerError("engine-run receipt identity contract changed")
        if receipt["bundle"] != {
            "expected_sha256_source": ENGINE_JOB_EXPECTED_SHA256_SOURCE,
            "hash_algorithm": ENGINE_JOB_BUNDLE_HASH_ALGORITHM,
            "schema": ENGINE_JOB_BUNDLE_SCHEMA,
        }:
            raise EngineRunnerError("engine-job bundle identity contract changed")
        run_identity = receipt["run_identity"]
        if run_identity != {
            "algorithm": ENGINE_RUN_ID_HASH_ALGORITHM,
            "domain": ENGINE_RUN_ID_DOMAIN,
            "sha256": run_identity.get("sha256") if type(run_identity) is dict else None,
        }:
            raise EngineRunnerError("receipt.run_identity contract changed")
        run_id = _digest(run_identity["sha256"], "receipt.run_identity.sha256")

        manifest_record = receipt["manifest"]
        artifact_record = receipt["trajectory_artifact"]
        archive_record = receipt["restart_archive"]
        if type(manifest_record) is not dict or set(manifest_record) != {
            "hash_algorithm",
            "schema",
            "scenario_id",
            "sha256",
            "size_bytes",
            "scope",
        }:
            raise EngineRunnerError("receipt.manifest key roster changed")
        if type(artifact_record) is not dict or set(artifact_record) != {
            "hash_algorithm",
            "schema",
            "sha256",
            "size_bytes",
            "scope",
        }:
            raise EngineRunnerError("receipt.trajectory_artifact key roster changed")
        if type(archive_record) is not dict or set(archive_record) != {
            "continuation_digest_algorithm",
            "continuation_digest_domain",
            "continuation_state_sha256",
            "hash_algorithm",
            "parent_integrator_id",
            "parent_native_result_digest",
            "parent_run_index",
            "schema",
            "sha256",
            "size_bytes",
            "scope",
        }:
            raise EngineRunnerError("receipt.restart_archive key roster changed")
        if (
            manifest_record["schema"] != PORTABLE_SCENARIO_MANIFEST_SCHEMA
            or manifest_record["scope"] != PORTABLE_SCENARIO_MANIFEST_SCOPE
            or manifest_record["hash_algorithm"]
            != PORTABLE_SCENARIO_HASH_ALGORITHM
            or manifest_record["scenario_id"] != scenario.scenario_id
            or manifest_record["sha256"] != expected_manifest_sha256
            or type(manifest_record["size_bytes"]) is not int
            or manifest_record["size_bytes"] != len(manifest_bytes)
        ):
            raise EngineRunnerError("receipt does not bind the supplied manifest")
        artifact_sha256 = _digest(
            artifact_record["sha256"], "receipt.trajectory_artifact.sha256"
        )
        archive_sha256 = _digest(
            archive_record["sha256"], "receipt.restart_archive.sha256"
        )
        if (
            artifact_record["schema"] != ENGINE_RUN_ARTIFACT_SCHEMA
            or artifact_record["scope"] != ENGINE_RUN_ARTIFACT_SCOPE
            or artifact_record["hash_algorithm"]
            != ENGINE_RUN_ARTIFACT_HASH_ALGORITHM
            or type(artifact_record["size_bytes"]) is not int
            or artifact_record["size_bytes"] != len(artifact_bytes)
            or engine_run_artifact_sha256(artifact_bytes) != artifact_sha256
        ):
            raise EngineRunnerError("receipt does not bind the trajectory artifact")
        if (
            archive_record["schema"] != SIMULATION_ARCHIVE_SCHEMA
            or archive_record["scope"] != SIMULATION_ARCHIVE_SCOPE
            or archive_record["hash_algorithm"]
            != SIMULATION_ARCHIVE_HASH_ALGORITHM
            or type(archive_record["size_bytes"]) is not int
            or archive_record["size_bytes"] != len(restart_archive_bytes)
            or simulation_archive_sha256(restart_archive_bytes) != archive_sha256
        ):
            raise EngineRunnerError("receipt does not bind the restart archive")

        artifact = load_engine_run_artifact(
            artifact_bytes,
            expected_sha256=artifact_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_run_id=run_id,
        )
        archive = load_simulation_archive(
            restart_archive_bytes,
            expected_sha256=archive_sha256,
            force_plan=scenario.force_plan,
        )
        force_plan_sha256 = simulation_force_plan_sha256(scenario.force_plan)
        if receipt["force_plan"] != {
            "digest_algorithm": FORCE_PLAN_DIGEST_ALGORITHM,
            "digest_domain": FORCE_PLAN_DIGEST_DOMAIN,
            "sha256": force_plan_sha256,
        }:
            raise EngineRunnerError("receipt force-plan identity mismatch")

        # Rebuild the canonical receipt from independently verified members.
        # The artifact owns the immutable run projection used here.
        projected_run = _projection_from_artifact(artifact)
        expected_document = _receipt_document(
            manifest_sha256=expected_manifest_sha256,
            manifest_size=len(manifest_bytes),
            scenario_id=scenario.scenario_id,
            force_plan_sha256=force_plan_sha256,
            run_id=run_id,
            run=projected_run,
            artifact_sha256=artifact_sha256,
            artifact_size=len(artifact_bytes),
            archive_sha256=archive_sha256,
            archive_size=len(restart_archive_bytes),
            continuation_state_sha256=(
                archive.continuation.continuation_state_sha256
            ),
            parent_run_index=archive.continuation.parent_run_index,
            parent_integrator_id=archive.continuation.parent_integrator_id,
            parent_native_result_digest=(
                archive.continuation.parent_native_result_digest
            ),
        )
        if receipt_bytes != _canonical(expected_document):
            raise EngineRunnerError("receipt is not the exact verified bundle projection")
        if _run_id(
            manifest_sha256=expected_manifest_sha256,
            force_plan_sha256=force_plan_sha256,
            run=projected_run,
        ) != run_id:
            raise EngineRunnerError("run_id is not derived from the verified run")

        expected_simulation_id = f"jx-engine-job-v1:{expected_manifest_sha256}"
        initial = scenario.initial_snapshot
        exact_coordinates = {
            "time_scale": initial.time_scale,
            "frame": initial.frame,
            "origin": initial.origin,
            "axes": initial.axes,
            "length_unit": initial.length_unit,
            "time_unit": initial.time_unit,
            "mass_unit": initial.mass_unit,
            "unit_system_id": initial.unit_system_id,
        }
        if (
            artifact.simulation_id != expected_simulation_id
            or artifact.run_id != run_id
            or artifact.run_index != 0
            or artifact.integrator_id != scenario.integration_spec.method_id
            or artifact.force_model_ids != scenario.force_model_ids
            or artifact.snapshot_id != initial.snapshot_id
            or artifact.plan_id != scenario.force_plan.plan_id
            or artifact.body_ids != initial.body_ids
            or artifact.backend_id != scenario.force_plan.backend.backend_id
            or artifact.device != scenario.force_plan.backend.device
            or any(
                getattr(artifact, field) != expected
                for field, expected in exact_coordinates.items()
            )
            or tuple(value.hex() for value in artifact.checkpoint_epochs)
            != tuple(
                float(value).hex()
                for value in scenario.integration_spec.checkpoint_epochs
            )
        ):
            raise EngineRunnerError(
                "trajectory artifact does not exactly represent the trusted manifest"
            )
        if (
            artifact.accepted_steps < artifact.checkpoint_count - 1
            or artifact.attempted_steps > scenario.integration_spec.maximum_steps
            or artifact.rejected_steps
            > scenario.integration_spec.maximum_rejections
            or artifact.force_evaluations
            != artifact.attempted_steps * RKF78_STAGE_COUNT
        ):
            raise EngineRunnerError(
                "trajectory artifact controller accounting is impossible"
            )

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - engine extra owns NumPy
            raise EngineRunnerError("Engine Job V1 verification requires NumPy") from exc

        def float_array_bytes(value: object) -> bytes:
            return np.ascontiguousarray(value, dtype=np.dtype("<f8")).tobytes(
                order="C"
            )

        def bool_array_bytes(value: object) -> bytes:
            return np.ascontiguousarray(value, dtype=np.bool_).tobytes(order="C")

        if not (
            float_array_bytes(artifact.checkpoint_positions[0])
            == float_array_bytes(initial.positions)
            and float_array_bytes(artifact.checkpoint_velocities[0])
            == float_array_bytes(initial.velocities)
        ):
            raise EngineRunnerError(
                "trajectory artifact initial state differs bitwise from the manifest"
            )

        continuation = archive.continuation
        restarted = continuation.snapshot
        expected_state_sha256 = continuation.continuation_state_sha256
        expected_snapshot_id = (
            f"{initial.snapshot_id}.continuation.0."
            f"{expected_state_sha256[:16]}"
        )
        expected_provenance = {
            "source_id": f"jx-continuation:{expected_simulation_id}:0",
            "citation": (
                "JX continuation of run 0 "
                f"({projected_run.integrator_id})"
            ),
            "version": "1",
            "sha256": expected_state_sha256,
        }
        if (
            continuation.simulation_id != projected_run.simulation_id
            or continuation.parent_run_index != projected_run.run_index
            or continuation.parent_integrator_id != projected_run.integrator_id
            or continuation.parent_native_result_digest
            != projected_run.native_result_digest
            or continuation.force_model_ids != projected_run.force_model_ids
            or restarted.epoch.hex() != projected_run.checkpoint_epochs[-1].hex()
            or restarted.body_ids != initial.body_ids
            or restarted.snapshot_id != expected_snapshot_id
            or any(
                getattr(restarted.provenance, field) != expected
                for field, expected in expected_provenance.items()
            )
            or any(
                getattr(restarted, field) != expected
                for field, expected in exact_coordinates.items()
            )
        ):
            raise EngineRunnerError(
                "restart archive is not the trajectory artifact endpoint"
            )
        if not (
            float_array_bytes(restarted.positions)
            == float_array_bytes(projected_run.checkpoint_positions[-1])
            and float_array_bytes(restarted.velocities)
            == float_array_bytes(projected_run.checkpoint_velocities[-1])
            and float_array_bytes(restarted.gravitational_parameters)
            == float_array_bytes(initial.gravitational_parameters)
            and float_array_bytes(restarted.masses)
            == float_array_bytes(initial.masses)
            and float_array_bytes(restarted.radii)
            == float_array_bytes(initial.radii)
            and bool_array_bytes(restarted.massive)
            == bool_array_bytes(initial.massive)
        ):
            raise EngineRunnerError(
                "restart archive endpoint arrays differ from the trajectory artifact"
            )
        return EngineRunVerification(
            run_id=run_id,
            manifest_sha256=expected_manifest_sha256,
            artifact_sha256=artifact_sha256,
            restart_archive_sha256=archive_sha256,
            receipt_sha256=expected_receipt_sha256,
            job_sha256=expected_job_sha256,
            manifest_bytes=manifest_bytes,
            job_bytes=_dump_job_bundle(
                artifact_bytes,
                restart_archive_bytes,
                receipt_bytes,
            ),
            artifact=artifact,
            archive=archive,
        )
    except EngineRunnerError:
        raise
    except (
        EngineRunArtifactError,
        ScenarioManifestError,
        SimulationArchiveError,
        JXSimulationContractError,
        KeyError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise EngineRunnerError(f"Engine Job V1 verification failed closed: {exc}") from exc


def verify_engine_job_bytes(
    manifest_bytes: bytes,
    *,
    expected_manifest_sha256: str,
    job_bytes: bytes,
    expected_job_sha256: str,
) -> EngineRunVerification:
    """Verify one externally hash-bound Engine Job V1 bundle."""

    expected_job_sha256 = _digest(expected_job_sha256, "expected_job_sha256")
    artifact_bytes, restart_archive_bytes, receipt_bytes = _load_job_bundle(
        job_bytes,
        expected_job_sha256,
    )
    return _verify_engine_run_components(
        manifest_bytes,
        expected_manifest_sha256=expected_manifest_sha256,
        artifact_bytes=artifact_bytes,
        restart_archive_bytes=restart_archive_bytes,
        receipt_bytes=receipt_bytes,
        expected_receipt_sha256=engine_run_receipt_sha256(receipt_bytes),
        expected_job_sha256=expected_job_sha256,
    )


def run_engine_job_bytes(
    manifest_bytes: bytes,
    *,
    expected_manifest_sha256: str,
) -> bytes:
    """Return the one deterministic outer bundle for a portable engine job."""

    return run_scenario_manifest(
        manifest_bytes,
        expected_manifest_sha256,
    ).job_bytes


__all__ = [
    "ENGINE_JOB_BUNDLE_HASH_ALGORITHM",
    "ENGINE_JOB_BUNDLE_MAX_BYTES",
    "ENGINE_JOB_BUNDLE_SCHEMA",
    "ENGINE_JOB_EXPECTED_SHA256_SOURCE",
    "ENGINE_JOB_V1_EXECUTION_POLICY",
    "ENGINE_JOB_V1_MAX_ARTIFACT_BYTES",
    "ENGINE_JOB_V1_MAX_BODIES",
    "ENGINE_JOB_V1_MAX_CHECKPOINTS",
    "ENGINE_JOB_V1_MAX_FORCE_MODELS",
    "ENGINE_JOB_V1_MAX_MANIFEST_BYTES",
    "ENGINE_JOB_V1_MAX_RESTART_BYTES",
    "ENGINE_JOB_V1_MAX_STAGE_BODY_PAIR_MODEL_EVALUATIONS",
    "ENGINE_JOB_V1_MAX_TEXT_ENCODING_BYTES",
    "ENGINE_JOB_V1_MAX_IDENTITY_ENCODING_BYTES",
    "ENGINE_JOB_V1_MAXIMUM_REJECTIONS",
    "ENGINE_JOB_V1_MAXIMUM_STEPS",
    "ENGINE_RUN_ID_DOMAIN",
    "ENGINE_RUN_ID_HASH_ALGORITHM",
    "ENGINE_RUN_RECEIPT_HASH_ALGORITHM",
    "ENGINE_RUN_RECEIPT_MAX_BYTES",
    "ENGINE_RUN_RECEIPT_SCHEMA",
    "ENGINE_RUN_RECEIPT_SCOPE",
    "EngineRunBundle",
    "EngineRunVerification",
    "EngineRunnerError",
    "engine_run_receipt_sha256",
    "engine_job_sha256",
    "engine_job_claim_ceiling",
    "run_engine_job_bytes",
    "run_scenario_manifest",
    "verify_engine_job_bytes",
]
