"""Evidence-aware registry for the JX General Dynamics research platform.

Registry membership means that a capability belongs to the same project.  It
does not mean that the capability is implemented in the Python engine, that it
is scientifically qualified, or that its physical model has been accepted.
Each row carries its own evidence state and claim ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class EvidenceState(str, Enum):
    QUALIFIED_BENCHMARK = "QUALIFIED_BENCHMARK"
    REPRODUCIBLE_SCREENING = "REPRODUCIBLE_SCREENING"
    INCONCLUSIVE = "INCONCLUSIVE"
    SUPPORTING_REFERENCE = "SUPPORTING_REFERENCE"
    BUILD_ONLY = "BUILD_ONLY"
    PLANNED = "PLANNED"


class IntegrationState(str, Enum):
    JX_NATIVE_BENCHMARK = "JX_NATIVE_BENCHMARK"
    PINNED_EXTERNAL_SOLVER = "PINNED_EXTERNAL_SOLVER"
    RESEARCH_BINDING_ONLY = "RESEARCH_BINDING_ONLY"
    PLANNED = "PLANNED"


@dataclass(frozen=True)
class GeneralDynamicsCapability:
    capability_id: str
    domain: str
    evidence_state: EvidenceState
    integration_state: IntegrationState
    evidence_path: str | None
    evidence_sha256: str | None
    claim_ceiling: str
    next_gate: str

    def __post_init__(self) -> None:
        for label in ("capability_id", "domain", "claim_ceiling", "next_gate"):
            value = getattr(self, label)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"{label} must be a nonempty trimmed string")
        if type(self.evidence_state) is not EvidenceState:
            raise TypeError("evidence_state must be EvidenceState")
        if type(self.integration_state) is not IntegrationState:
            raise TypeError("integration_state must be IntegrationState")
        if self.evidence_state is EvidenceState.PLANNED:
            if self.evidence_path is not None or self.evidence_sha256 is not None:
                raise ValueError("planned capability cannot claim result evidence")
        else:
            if type(self.evidence_path) is not str or not self.evidence_path:
                raise ValueError("non-planned capability requires evidence_path")
            evidence_path = Path(self.evidence_path)
            if (
                evidence_path.is_absolute()
                or not evidence_path.parts
                or any(part in (".", "..") for part in evidence_path.parts)
            ):
                raise ValueError(
                    "evidence_path must be a repository-relative path without traversal"
                )
            digest = self.evidence_sha256
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("non-planned capability requires lowercase SHA-256")

    @property
    def scientifically_qualified(self) -> bool:
        return self.evidence_state is EvidenceState.QUALIFIED_BENCHMARK


_CAPABILITIES = (
    GeneralDynamicsCapability(
        "orbit.equal_binary",
        "ORBITAL_DYNAMICS",
        EvidenceState.QUALIFIED_BENCHMARK,
        IntegrationState.JX_NATIVE_BENCHMARK,
        "runs/jx_equal_binary_100_period_repro_v1/reference_science_v1.json",
        "7fa6f88a817738d88da18f74636dfb5223232e0a718aad6853a07ba8bc4ea2d6",
        "THE_FROZEN_NEWTONIAN_EQUAL_BINARY_WORKLOAD_ONLY",
        "INDEPENDENT_EXTERNAL_REPRODUCTION_BEFORE_BROADER_ORBITAL_CLAIMS",
    ),
    GeneralDynamicsCapability(
        "orbit.de440_reduced_eleven_body",
        "ORBITAL_DYNAMICS",
        EvidenceState.REPRODUCIBLE_SCREENING,
        IntegrationState.JX_NATIVE_BENCHMARK,
        "runs/jx_gpu_scientific_ladder_v1/evidence/eleven/normal_a/report.json",
        "576876db07c15b4e62047c615672524a046ce4484021799a2ee9a7353ad00b27",
        "ONE_DAY_CPU_GPU_PARITY_FOR_THE_FROZEN_REDUCED_MODEL",
        "INDEPENDENT_INTEGRATOR_AND_COMPLETE_SAME_MODEL_CONVERGENCE_LADDER",
    ),
    GeneralDynamicsCapability(
        "orbit.solar_system_linear_secular_4p5gyr",
        "ORBITAL_DYNAMICS",
        EvidenceState.INCONCLUSIVE,
        IntegrationState.JX_NATIVE_BENCHMARK,
        "runs/jx_gpu_de440_multi_epoch_long_ladder_v1/SUMMARY.json",
        "b171bcdea63f72070dd6f1593853ffa9dadb15e7ea09d52ed7ec4dcffc23ff33",
        "SIXTY_THREE_EPOCH_100YR_GPU_STRESS_PASS_WITH_STOPPED_THREE_ANCHOR_1000YR_REFERENCE_LADDER_NO_EPHEMERIS_OR_STABILITY_CLAIM",
        "PREREGISTER_600_SECOND_FOURTH_SYMPLECTIC_LEVEL_WITH_REFERENCE_AND_EXISTING_GATES_UNCHANGED",
    ),
    GeneralDynamicsCapability(
        "rotation.lunar_mantle_core",
        "ROTATIONAL_AND_INTERIOR_DYNAMICS",
        EvidenceState.INCONCLUSIVE,
        IntegrationState.RESEARCH_BINDING_ONLY,
        "runs/jx_lunar_deformation_geodetic_screen_v1/SUMMARY.json",
        "3c09b91a60f9f4b0ffa16596c335f0a422a6122ceaae26e62458e2f317be9d95",
        "COMPLETE_DEFORMATION_AND_GEODETIC_PROMOTION_GATES_FAILED_NO_LLR_OR_DE440_REPRODUCTION_CLAIM",
        "PREREGISTER_SIMULTANEOUS_TRANSLATION_ROTATION_WITH_COMPLETE_DELAYED_DEFORMATION_AND_COMMON_POTENTIAL_REACTIONS",
    ),
    GeneralDynamicsCapability(
        "cosmology.collisionless_particle_mesh",
        "COSMOLOGICAL_DYNAMICS",
        EvidenceState.QUALIFIED_BENCHMARK,
        IntegrationState.JX_NATIVE_BENCHMARK,
        "runs/jx_cosmology_pm_256_independent_r2/REPORT.json",
        "86fdb8ceaddf2067de8f296d3bd55e1dde23e4a9b58e6ecf3bf4b02c7179a014",
        "FROZEN_256_CUBED_COLLISIONLESS_EDS_GROWING_MODE_WITH_INDEPENDENT_PLANAR_CPU_COMPARISON_NO_PRODUCTION_COSMOLOGY_CLAIM",
        "TEST_ARBITRARY_3D_PERTURBATIONS_AGAINST_AN_INDEPENDENTLY_MAINTAINED_EXTERNAL_PARTICLE_MESH_CODE",
    ),
    GeneralDynamicsCapability(
        "cosmology.dark_matter_baryon_3d",
        "COSMOLOGICAL_HYDRODYNAMICS",
        EvidenceState.QUALIFIED_BENCHMARK,
        IntegrationState.JX_NATIVE_BENCHMARK,
        "runs/jx_cosmology_baryons_3d_r4/REPORT.json",
        "d1b70af737b7d85943c4e73b884c51330e17b94f5a3427386d7959aaaa56c499",
        "FROZEN_128_CUBED_EDS_SEPARABLE_THREE_MODE_DARK_MATTER_BARYON_VALIDATION_WITH_PHASE_SENSITIVE_GATES_NO_PRODUCTION_COSMOLOGY_CLAIM",
        "COMPARE_A_NONLINEAR_OBLIQUE_3D_DARK_MATTER_GAS_CASE_WITH_AN_INDEPENDENTLY_MAINTAINED_EXTERNAL_COSMOLOGY_CODE_AT_MULTIPLE_RESOLUTIONS",
    ),
    GeneralDynamicsCapability(
        "hydro.ideal_gas_shocks",
        "COMPRESSIBLE_HYDRODYNAMICS",
        EvidenceState.QUALIFIED_BENCHMARK,
        IntegrationState.PINNED_EXTERNAL_SOLVER,
        "runs/jx_sn_hydro_01_final_caps/FINAL.json",
        "05f9e6fff6e5afe8f44d6c86f168410074e36f82401e9b454e6ffb586d0110df",
        "FROZEN_CASTRO_SOD_AND_SEDOV_CPU_GPU_BENCHMARKS_ONLY",
        "REACTIVE_SOURCE_AND_HYDRO_COUPLING_QUALIFICATION",
    ),
    GeneralDynamicsCapability(
        "hydro.reacting_nuclear_detonation",
        "REACTING_HYDRODYNAMICS",
        EvidenceState.INCONCLUSIVE,
        IntegrationState.PINNED_EXTERNAL_SOLVER,
        "runs/jx_reacting_hydrodynamics_reality_v1/RESULT.json",
        "7e702a64df75e5c8b5c7ea8243a228162a1cf927be87241d313ec58afa54f1ab",
        "CJ_SPEED_REALITY_TEST_FAILED_SIMULATION_SPEEDS_33P6_PERCENT_BELOW_IDEAL_REFERENCE",
        "DIAGNOSE_FRONT_OBSERVABLE_AND_STEADY_STATE_BEFORE_MULTIDIMENSIONAL_RUN",
    ),
    GeneralDynamicsCapability(
        "thermochemistry.air5",
        "HIGH_TEMPERATURE_GAS_PHYSICS",
        EvidenceState.SUPPORTING_REFERENCE,
        IntegrationState.PINNED_EXTERNAL_SOLVER,
        "runs/jx_reacting_aerodynamics_air5_readiness_v1/REPORT.json",
        "df6b7c91971b5ae0aac96e5f1c953a8a369125dbd8e3d6fa9f3e5b65a34c8368",
        "THERMOCHEMISTRY_AND_SOLVER_READINESS_NO_QUALIFIED_FLOW",
        "POSITIVITY_PRESERVING_SOURCE_AND_REACTING_SHOCK_REFINEMENT",
    ),
    GeneralDynamicsCapability(
        "hydro.radiation_reacting_multiphysics",
        "RADIATION_REACTING_HYDRODYNAMICS",
        EvidenceState.PLANNED,
        IntegrationState.PLANNED,
        None,
        None,
        "NO_IMPLEMENTATION_OR_SCIENTIFIC_RESULT",
        "QUALIFY_REACTING_HYDRODYNAMICS_BEFORE_ADDING_RADIATION",
    ),
)


GENERAL_DYNAMICS_CAPABILITIES: tuple[GeneralDynamicsCapability, ...] = _CAPABILITIES
GENERAL_DYNAMICS_BY_ID: Mapping[str, GeneralDynamicsCapability] = MappingProxyType(
    {row.capability_id: row for row in _CAPABILITIES}
)

if len(GENERAL_DYNAMICS_BY_ID) != len(_CAPABILITIES):
    raise RuntimeError("duplicate JX General Dynamics capability identifier")


def capability(capability_id: str) -> GeneralDynamicsCapability:
    """Return one exact registry row, failing closed on an unknown identifier."""

    if type(capability_id) is not str:
        raise TypeError("capability_id must be str")
    try:
        return GENERAL_DYNAMICS_BY_ID[capability_id]
    except KeyError as exc:
        raise KeyError(f"unknown JX General Dynamics capability: {capability_id}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evidence(repository_root: Path) -> dict[str, bool]:
    """Verify every evidence-bearing row against its frozen file digest."""

    root = repository_root.resolve(strict=True)
    results: dict[str, bool] = {}
    for row in _CAPABILITIES:
        if row.evidence_path is None:
            results[row.capability_id] = row.evidence_state is EvidenceState.PLANNED
            continue
        relative = Path(row.evidence_path)
        candidate = root
        unsafe_component = False
        for component in relative.parts:
            candidate = candidate / component
            if candidate.is_symlink():
                unsafe_component = True
                break
        if unsafe_component:
            results[row.capability_id] = False
            continue
        try:
            path = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            results[row.capability_id] = False
            continue
        if not path.is_relative_to(root) or not path.is_file():
            results[row.capability_id] = False
            continue
        try:
            results[row.capability_id] = _sha256(path) == row.evidence_sha256
        except OSError:
            results[row.capability_id] = False
    return results


def registry_summary() -> dict[str, object]:
    """Return a serialization-safe summary without promoting partial evidence."""

    counts = {state.value: 0 for state in EvidenceState}
    rows: list[dict[str, object]] = []
    for row in _CAPABILITIES:
        counts[row.evidence_state.value] += 1
        rows.append(
            {
                "capability_id": row.capability_id,
                "claim_ceiling": row.claim_ceiling,
                "domain": row.domain,
                "evidence_path": row.evidence_path,
                "evidence_sha256": row.evidence_sha256,
                "evidence_state": row.evidence_state.value,
                "integration_state": row.integration_state.value,
                "next_gate": row.next_gate,
                "scientifically_qualified": row.scientifically_qualified,
            }
        )
    return {
        "schema": "jxplanetx.general-dynamics-capability-registry.v1",
        "project": "JX_GENERAL_DYNAMICS",
        "version": "0.6.0rc3",
        "overall_status": "PARTIALLY_QUALIFIED_RESEARCH_PLATFORM",
        "scientific_claim_state": "SCREENING_ONLY",
        "production_ready": False,
        "unified_multiphysics_execution": False,
        "counts": counts,
        "capabilities": rows,
    }
