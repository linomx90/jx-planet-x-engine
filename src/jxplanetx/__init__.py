"""JX Scientific Engine: falsification-first dynamics and evidence framework."""

from __future__ import annotations

from typing import Any

__version__ = "0.6.0rc17"

_GR15_EXPORTS = (
    "GR15ContractError",
    "GR15Error",
    "GR15IntegrationError",
    "GR15Result",
    "GR15Spec",
    "GR15UnavailableError",
    "GR15Workspace",
    "gr15_runtime_identity",
    "integrate_gr15",
    "prepare_gr15_workspace",
)

_GR15_EIH_1PN_EXPORTS = (
    "GR15EIH1PNIntegrationError",
    "GR15EIH1PNResult",
    "GR15EIH1PNWorkspace",
    "gr15_eih_1pn_runtime_identity",
    "integrate_gr15_eih_1pn",
    "prepare_gr15_eih_1pn_workspace",
)

_EIH_1PN_CUDA_EXPORTS = (
    "EIH1PNCUDABatchResult",
    "EIH1PNCUDAContractError",
    "EIH1PNCUDAError",
    "EIH1PNCUDAEvaluationError",
    "EIH1PNCUDAUnavailableError",
    "GR15EIH1PNCUDAStageResult",
    "eih_1pn_cuda_runtime_identity",
    "evaluate_eih_1pn_total_acceleration_cuda",
    "evaluate_gr15_eih_1pn_stages_cuda",
)

_GR15_EIH_1PN_CUDA_EXPORTS = (
    "GR15EIH1PNCUDABatchResult",
    "GR15EIH1PNCUDAContractError",
    "GR15EIH1PNCUDAError",
    "GR15EIH1PNCUDAIntegrationError",
    "GR15EIH1PNCUDASystemAudit",
    "gr15_eih_1pn_cuda_runtime_identity",
    "integrate_gr15_eih_1pn_cuda_batch",
)

_GR15_EIH_1PN_BATCH_EXPORTS = (
    "AUTO_API_ID",
    "CPU_BACKEND_ID",
    "CUDA_BACKEND_ID",
    "DEFAULT_CPU_WORKERS",
    "DEFAULT_DISPATCH_POLICY",
    "GR15EIH1PNBatchContractError",
    "GR15EIH1PNBatchError",
    "GR15EIH1PNBatchIntegrationError",
    "GR15EIH1PNBatchResult",
    "GR15EIH1PNDispatchDecision",
    "GR15EIH1PNDispatchPolicy",
    "GR15EIH1PNSystemAudit",
    "RTX_5060_TI_TEN_YEAR_ELEVEN_BODY_POLICY",
    "integrate_gr15_eih_1pn_batch",
    "integrate_gr15_eih_1pn_cpu_batch",
    "select_gr15_eih_1pn_backend",
)

_GR15_EIH_1PN_VERIFIED_EXPORTS = (
    "GR15EIH1PNOperationalContext",
    "GR15EIH1PNVerificationContractError",
    "GR15EIH1PNVerificationError",
    "GR15EIH1PNVerificationFailure",
    "GR15EIH1PNVerificationGates",
    "GR15EIH1PNVerifiedResult",
    "integrate_verified_gr15_eih_1pn",
)

_UNIFIED_SIMULATION_EXPORTS = (
    "COUPLED_LUNAR_BODY_IDS",
    "COUPLED_LUNAR_FORCE_DISPATCH_SCOPE",
    "COUPLED_LUNAR_RESULT_DIGEST_ALGORITHM",
    "COUPLED_LUNAR_RESULT_DIGEST_DOMAIN",
    "COUPLED_LUNAR_RKF78_METHOD_ID",
    "COUPLED_LUNAR_STATE_ABI_VERSION",
    "CONTINUATION_DIGEST_ALGORITHM",
    "CONTINUATION_DIGEST_DOMAIN",
    "CoupledLunarContractError",
    "CoupledLunarRun",
    "CoupledLunarStateSnapshot",
    "EARTH_ZONAL_PARAMETER_IDS",
    "EXACT_ORDERED_FORCE_MODEL_IDS_SCOPE",
    "FORCE_ABI_REGISTRY",
    "FORCE_ABI_VERSION",
    "FORCE_ABI_V1_GENERAL_FIRST_ORDER_SCOPE",
    "ForceABISpec",
    "FORCE_PLAN_DIGEST_ALGORITHM",
    "FORCE_PLAN_DIGEST_DOMAIN",
    "GR15_EIH1PN_CUDA_METHOD_ID",
    "GR15_EIH1PN_METHOD_ID",
    "GR15_NEWTONIAN_METHOD_ID",
    "INTEGRATOR_ROUTE_REGISTRY",
    "INTEGRATOR_ROUTE_REGISTRY_VERSION",
    "INTEGRATOR_ROUTE_SCIENTIFIC_CLAIM_STATE",
    "IntegratorRouteSpec",
    "JXSimulation",
    "JXSimulationArchiveLoad",
    "JXSimulationCompatibilityError",
    "JXSimulationContinuation",
    "JXSimulationContractError",
    "JXSimulationError",
    "JXSimulationRun",
    "LUNAR_DEGREE2_PARAMETER_IDS",
    "LUNAR_DEGREE3_PARAMETER_IDS",
    "LUNAR_EPHEMERIS_V1_BODY_IDS",
    "LUNAR_EPHEMERIS_V1_FORCE_ROSTER",
    "LUNAR_EPHEMERIS_V1_METHOD_ID",
    "LUNAR_EPHEMERIS_V1_MODEL_ID",
    "LUNAR_EPHEMERIS_V1_OMITTED_PHYSICS",
    "LUNAR_EPHEMERIS_V1_RESULT_DIGEST_DOMAIN",
    "LUNAR_EPHEMERIS_V1_STATE_ABI_VERSION",
    "LunarEphemerisV1Error",
    "LunarEphemerisV1IntegrationSpec",
    "LunarEphemerisV1Parameters",
    "LunarEphemerisV1Run",
    "LunarEphemerisV1State",
    "LunarEphemerisV1StepLimitError",
    "EarthZonalJ2J5Force",
    "LunarStaticDegree2Force",
    "LunarStaticDegree3Force",
    "SIMULATION_ARCHIVE_HASH_ALGORITHM",
    "SIMULATION_ARCHIVE_MAX_BYTES",
    "SIMULATION_ARCHIVE_SCHEMA",
    "SIMULATION_ARCHIVE_SCOPE",
    "STATE_BLOCK_REGISTRY",
    "SimulationArchiveError",
    "StateBlockSpec",
    "MutualEIH1PN",
    "bind_earth_zonal_j2_j5_force",
    "bind_coupled_lunar_state",
    "bind_lunar_static_degree2_force",
    "bind_lunar_static_degree3_force",
    "dump_simulation_archive",
    "get_force_abi",
    "get_integrator_route",
    "get_state_block",
    "integrate_deformable_coupled_lunar_state",
    "integrate_lunar_ephemeris_v1",
    "list_force_abis",
    "list_integrator_routes",
    "list_state_blocks",
    "load_simulation_archive",
    "simulation_archive_sha256",
    "simulation_force_plan_sha256",
)

_ENGINE_JOB_EXPORTS = (
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
    "ENGINE_RUN_ARTIFACT_ARRAY_DTYPE",
    "ENGINE_RUN_ARTIFACT_HASH_ALGORITHM",
    "ENGINE_RUN_ARTIFACT_MAX_BYTES",
    "ENGINE_RUN_ARTIFACT_MAX_MANIFEST_BYTES",
    "ENGINE_RUN_ARTIFACT_MEMBER_HASH_ALGORITHM",
    "ENGINE_RUN_ARTIFACT_SCHEMA",
    "ENGINE_RUN_ARTIFACT_SCOPE",
    "ENGINE_RUN_ID_DOMAIN",
    "ENGINE_RUN_ID_HASH_ALGORITHM",
    "ENGINE_RUN_RECEIPT_HASH_ALGORITHM",
    "ENGINE_RUN_RECEIPT_MAX_BYTES",
    "ENGINE_RUN_RECEIPT_SCHEMA",
    "ENGINE_RUN_RECEIPT_SCOPE",
    "EngineRunArtifactError",
    "EngineRunBundle",
    "EngineRunVerification",
    "EngineRunnerError",
    "JXEngineRunArtifact",
    "SIMULATION_ARCHIVE_MAX_MANIFEST_BYTES",
    "dump_engine_run_artifact",
    "engine_job_sha256",
    "engine_job_claim_ceiling",
    "engine_run_artifact_sha256",
    "engine_run_receipt_sha256",
    "load_engine_run_artifact",
    "run_engine_job_bytes",
    "run_scenario_manifest",
    "verify_engine_job_bytes",
)


def __getattr__(name: str) -> Any:
    """Load the NumPy GR15 surface only when a caller requests it."""

    if name in _GR15_EXPORTS:
        from . import gr15

        value = getattr(gr15, name)
        globals()[name] = value
        return value
    if name in _GR15_EIH_1PN_EXPORTS:
        from . import gr15_eih_1pn

        value = getattr(gr15_eih_1pn, name)
        globals()[name] = value
        return value
    if name in _EIH_1PN_CUDA_EXPORTS:
        from . import eih_1pn_cuda

        value = getattr(eih_1pn_cuda, name)
        globals()[name] = value
        return value
    if name in _GR15_EIH_1PN_CUDA_EXPORTS:
        from . import gr15_eih_1pn_cuda

        value = getattr(gr15_eih_1pn_cuda, name)
        globals()[name] = value
        return value
    if name in _GR15_EIH_1PN_BATCH_EXPORTS:
        from . import gr15_eih_1pn_batch

        value = getattr(gr15_eih_1pn_batch, name)
        globals()[name] = value
        return value
    if name in _GR15_EIH_1PN_VERIFIED_EXPORTS:
        from . import gr15_eih_1pn_verified

        value = getattr(gr15_eih_1pn_verified, name)
        globals()[name] = value
        return value
    if name in _UNIFIED_SIMULATION_EXPORTS or name in _ENGINE_JOB_EXPORTS:
        from . import engine

        value = getattr(engine, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    *_GR15_EXPORTS,
    *_GR15_EIH_1PN_EXPORTS,
    *_EIH_1PN_CUDA_EXPORTS,
    *_GR15_EIH_1PN_CUDA_EXPORTS,
    *_GR15_EIH_1PN_BATCH_EXPORTS,
    *_GR15_EIH_1PN_VERIFIED_EXPORTS,
    *_UNIFIED_SIMULATION_EXPORTS,
    *_ENGINE_JOB_EXPORTS,
]
