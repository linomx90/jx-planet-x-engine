import dataclasses
import hashlib
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import jxplanetx.engine.encounter as encounter_runtime
import jxplanetx.engine.hybrid as hybrid_runtime
import jxplanetx.engine.wisdom_holman as wh_runtime
from jxplanetx.engine.hybrid import (
    HybridContractError,
    HybridLaneExecutionCounts,
    HybridOuterStepRecord,
    HybridProbeWorkTotals,
    HybridWisdomHolmanDiagnostics,
    HybridWisdomHolmanRKF78Result,
    integrate_hybrid_wisdom_holman_rkf78_trajectory,
)
from jxplanetx.engine.hybrid_contracts import (
    HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER,
    HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER,
    HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER,
    HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER,
    HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER,
    HybridFarProbeDecision,
    HybridEncounterControlProfile,
    HybridKeplerProbeRecord,
    HybridPrivateEncounterRecord,
    HybridWisdomHolmanRKF78Spec,
    _bind_encounter_segment,
)
from jxplanetx.engine.trajectory import TrajectoryDomainError
from jxplanetx.engine.wisdom_holman import integrate_wisdom_holman_trajectory

from tests.test_engine_encounter import _resources
from tests.test_engine_wisdom_holman import (
    circular_planet_state,
    runtime_plan,
    runtime_spec,
)


def encounter_profile(
    *,
    fixed_step: float,
    floor: float = 0.01,
    atol: float = 1.0e-6,
    **changes: object,
) -> HybridEncounterControlProfile:
    magnitude = abs(fixed_step)
    values: dict[str, object] = {
        "initial_step_magnitude": magnitude,
        "minimum_step_magnitude": 0.001,
        "maximum_step_magnitude": magnitude,
        "pair_certification_floors": (floor,),
        "pair_position_atols": (atol,),
        "pair_position_rtol": 1.0e-12,
        "pair_velocity_atols": (atol,),
        "pair_velocity_rtol": 1.0e-12,
        "gm_centroid_position_atol": atol,
        "gm_centroid_position_rtol": 1.0e-12,
        "gm_centroid_velocity_atol": atol,
        "gm_centroid_velocity_rtol": 1.0e-12,
        "maximum_substep_proposals": 128,
        "maximum_accepted_substeps": 128,
        "maximum_rejected_substeps": 64,
        "maximum_consecutive_rejections": 32,
        "maximum_force_evaluations": 1664,
        "safety_factor": 0.9,
        "minimum_scale_factor": 0.2,
        "maximum_scale_factor": 5.0,
        "exact_rational_resources": _resources(),
    }
    values.update(changes)
    return HybridEncounterControlProfile(**values)  # type: ignore[arg-type]


def hybrid_fixture(
    *,
    steps: int = 2,
    fixed_step: float = 0.125,
    checkpoints: tuple[int, ...] | None = None,
    wh_floor: float = 0.01,
    periapse_floor: float = 0.05,
    child_floor: float = 0.01,
    atol: float = 1.0e-6,
):
    state = circular_planet_state()
    wh_spec = runtime_spec(
        state,
        final_step=steps,
        fixed_step=fixed_step,
        checkpoints=checkpoints or (0, steps),
        minimum_encounter_pair_separation=wh_floor,
        minimum_jacobi_periapse=periapse_floor,
    )
    profile = encounter_profile(
        fixed_step=fixed_step, floor=child_floor, atol=atol
    )
    return (
        state,
        runtime_plan(state),
        HybridWisdomHolmanRKF78Spec(wh_spec, profile),
    )


_PRIVATE_COMPONENT_LITERAL_KATS = {
    'final_state_content_sha256': (b'jxplanetx.hybrid-private-encounter-final-state.content-integrity.v1\x00{"positions":{"dtype":"float64","shape":[2,3],"values":["-0x1.03d6379884931p-10","-0x1.05549f125d7b2p-13","0x0.0p+0","0x1.fb7e6495e2ef3p-1","0x1.fe6946afde946p-4","0x0.0p+0"]},"schema":"jxplanetx.hybrid-private-encounter-final-state.payload.v1","velocities":{"dtype":"float64","shape":[2,3],"values":["0x1.05761036267a5p-13","-0x1.03f777ccd02cap-10","0x0.0p+0","-0x1.feaa97a9c326ep-4","0x1.fbbf55fc06972p-1","0x0.0p+0"]}}', 489, '816c7feb2cad928c1609a2c7a6cdd9cde1eb377347b01b80391e0be8da72c4e4'),
    'initialization_content_sha256': (b'jxplanetx.hybrid-private-encounter-initialization.content-integrity.v1\x00{"record":{"dataclass":"jxplanetx.engine.encounter.EncounterInitializationRecord","fields":{"dyadic_operations":400,"exact_operations":921,"gcd_iterations":80,"maximum_integer_bits":169,"maximum_rational_exponent_magnitude":124,"rational_operations":521,"transcript_bytes":1863,"witness_content_sha256":"24009e965695a525d8a94dd4db2c06cb8b04220bf691237dc278344b41cf4fdf"}},"schema":"jxplanetx.hybrid-private-encounter-initialization.payload.v1"}', 515, 'fc818f80229f200f1d6db6d03f3020b3cf6e3336fff9536c0d90a82e65a5b66e'),
    'proposal_ledger_content_sha256': (b'jxplanetx.hybrid-private-encounter-proposal-ledger.content-integrity.v1\x00jxplanetx.hybrid-private-encounter-proposal-ledger.sequence.v1\x00\x00\x00\x00\x00\x00\x00\x00(\x00\x00\x00\x00\x00\x00\x027{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":471,"exact_operations":2453,"gcd_iterations":736,"index":1,"maximum_integer_bits":465,"maximum_rational_exponent_magnitude":55,"normalized_error":null,"rational_operations":1982,"signed_step":{"float_hex":"0x1.0000000000000p-3"},"terminal_below_minimum":false,"transcript_bytes":3952,"witness_content_sha256":"45e5f98a2bced0ba0d6aa73a10bd741612edb1d811d89ac4d7cf32ee24ae2fa4"}}\x00\x00\x00\x00\x00\x00\x027{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":471,"exact_operations":2453,"gcd_iterations":743,"index":2,"maximum_integer_bits":461,"maximum_rational_exponent_magnitude":56,"normalized_error":null,"rational_operations":1982,"signed_step":{"float_hex":"0x1.0000000000000p-4"},"terminal_below_minimum":false,"transcript_bytes":3950,"witness_content_sha256":"d096343e771bab1a5923572f86ae7c553ab8ba5efaef1d66f9b42d1cbb53273f"}}\x00\x00\x00\x00\x00\x00\x027{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":471,"exact_operations":2453,"gcd_iterations":709,"index":3,"maximum_integer_bits":457,"maximum_rational_exponent_magnitude":57,"normalized_error":null,"rational_operations":1982,"signed_step":{"float_hex":"0x1.0000000000000p-5"},"terminal_below_minimum":false,"transcript_bytes":3948,"witness_content_sha256":"1613cc3202887a0e151b40fbec653c0af56fe005b345b5de7560a042874afb7d"}}\x00\x00\x00\x00\x00\x00\x027{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":471,"exact_operations":2453,"gcd_iterations":738,"index":4,"maximum_integer_bits":453,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":1982,"signed_step":{"float_hex":"0x1.0000000000000p-6"},"terminal_below_minimum":false,"transcript_bytes":3947,"witness_content_sha256":"5e84803446e98983e078e02689815144c1fd8ea4fbd872e137fe4e12935af1bf"}}\x00\x00\x00\x00\x00\x00\x02M{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3702,"exact_operations":5684,"gcd_iterations":756,"index":5,"maximum_integer_bits":452,"maximum_rational_exponent_magnitude":146,"normalized_error":{"float_hex":"0x1.e9b635bfbfbc5p-45"},"rational_operations":1982,"signed_step":{"float_hex":"0x1.0000000000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7212,"witness_content_sha256":"e94ef5e40d7382359da6fd1ae959edbe4590d5d30d0b9a2464e1929c4ef3c56f"}}\x00\x00\x00\x00\x00\x00\x028{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1722,"index":6,"maximum_integer_bits":463,"maximum_rational_exponent_magnitude":64,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.4000000000000p-5"},"terminal_below_minimum":false,"transcript_bytes":4028,"witness_content_sha256":"ceaf3305f16bc44b3c9100a7a18468655d305410b0ea3af311af7e1428822130"}}\x00\x00\x00\x00\x00\x00\x028{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1736,"index":7,"maximum_integer_bits":459,"maximum_rational_exponent_magnitude":64,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.4000000000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4027,"witness_content_sha256":"a013301b8c49a370f40a520e032965dd966ed6b23593a4335c6ef62cd883746e"}}\x00\x00\x00\x00\x00\x00\x02N{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1726,"index":8,"maximum_integer_bits":457,"maximum_rational_exponent_magnitude":138,"normalized_error":{"float_hex":"0x1.312cffa5ea264p-44"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.4000000000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7290,"witness_content_sha256":"ba1fad62d407c52db059ba091212cc21258bb3627f8a446e0562e57c19d2b026"}}\x00\x00\x00\x00\x00\x00\x028{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1780,"index":9,"maximum_integer_bits":462,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.9000000000000p-5"},"terminal_below_minimum":false,"transcript_bytes":4029,"witness_content_sha256":"6845e36480ea43fb5c9100951c2ed85d9d16779e0ffd0652acf1dbacad90a689"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1793,"index":10,"maximum_integer_bits":458,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.9000000000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4036,"witness_content_sha256":"19bc1e2fb6761978dca93872ac1976829c9b6358f103aa9d793eb16cc595707e"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1762,"index":11,"maximum_integer_bits":455,"maximum_rational_exponent_magnitude":136,"normalized_error":{"float_hex":"0x1.7dbfc5cd12a89p-42"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.9000000000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7287,"witness_content_sha256":"e3709bcd987a2e90af5406e8112706c8488a4468a7304557afe382911b4fbbc3"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1776,"index":12,"maximum_integer_bits":461,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.f400000000000p-5"},"terminal_below_minimum":false,"transcript_bytes":4050,"witness_content_sha256":"df3b572052c238449292f76dded922dc82a02408a7a2f38bb95cac1fcdc2301b"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1787,"index":13,"maximum_integer_bits":457,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.f400000000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4049,"witness_content_sha256":"a99955b30bf8b7cf6734ce5c73dbdd6113fb1ee20459c4804578db8cfc04e0fc"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1779,"index":14,"maximum_integer_bits":453,"maximum_rational_exponent_magnitude":59,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.f400000000000p-7"},"terminal_below_minimum":false,"transcript_bytes":4047,"witness_content_sha256":"d5fde4a21d651be9e3846d5db708571d767bc7dc87e3bf08a4e003967154cf8f"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1796,"index":15,"maximum_integer_bits":453,"maximum_rational_exponent_magnitude":134,"normalized_error":{"float_hex":"0x1.dc9ab40a99bbap-44"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.f400000000000p-8"},"terminal_below_minimum":false,"transcript_bytes":7289,"witness_content_sha256":"0ee84bee10d91746cdee5c79c1042e144c058c5581c8228daa484642ae57e23f"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1805,"index":16,"maximum_integer_bits":452,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.3880000000000p-5"},"terminal_below_minimum":false,"transcript_bytes":4040,"witness_content_sha256":"ad5bb195a7686c9ec7d9f9d74c6983e8e181db8e30fb2b456097ce93960131e7"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1821,"index":17,"maximum_integer_bits":448,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.3880000000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4039,"witness_content_sha256":"b9e8480f9d9e13cad11f1e57e2dab36f90b3490e8988de04426436d32c9caf53"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1819,"index":18,"maximum_integer_bits":447,"maximum_rational_exponent_magnitude":134,"normalized_error":{"float_hex":"0x1.2a5072917bb82p-44"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.3880000000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7286,"witness_content_sha256":"f264f748510f21351ef47f64fa632dfda916824fc93e2d45edf839e37b82a87f"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1806,"index":19,"maximum_integer_bits":460,"maximum_rational_exponent_magnitude":57,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.86a0000000000p-5"},"terminal_below_minimum":false,"transcript_bytes":4058,"witness_content_sha256":"33d64cd4b22db15104b1f9ce577410a6fa3ddfc7dd12b3ef4524165412251065"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1821,"index":20,"maximum_integer_bits":456,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.86a0000000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4057,"witness_content_sha256":"3f3c662e34bc5c42c3ea18e3d55316db3e320e04b5c256394e6c48a92dffe4e2"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1806,"index":21,"maximum_integer_bits":453,"maximum_rational_exponent_magnitude":134,"normalized_error":{"float_hex":"0x1.74b5fdfd86c09p-43"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.86a0000000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7303,"witness_content_sha256":"ead9c10961863fdf6da89a33b2ad936feb677108677e557b981bc027356ca5e9"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1822,"index":22,"maximum_integer_bits":459,"maximum_rational_exponent_magnitude":57,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.e848000000000p-5"},"terminal_below_minimum":false,"transcript_bytes":4064,"witness_content_sha256":"dd3a0c1dad83b0e01e09829248ae283961ce83ece9539f7515dd2e76f0a0382c"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1822,"index":23,"maximum_integer_bits":455,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.e848000000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4063,"witness_content_sha256":"73125247f2484d993d4e922d3da45dcfe29802ac9c9efba5c95cc07f70edfe20"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1791,"index":24,"maximum_integer_bits":451,"maximum_rational_exponent_magnitude":59,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.e848000000000p-7"},"terminal_below_minimum":false,"transcript_bytes":4061,"witness_content_sha256":"48939524aa438ac9370cdd59d03e99173d5fbba7df8b1193fb3a777f9168209a"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1800,"index":25,"maximum_integer_bits":451,"maximum_rational_exponent_magnitude":134,"normalized_error":{"float_hex":"0x1.d134ddc7b0bedp-45"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.e848000000000p-8"},"terminal_below_minimum":false,"transcript_bytes":7303,"witness_content_sha256":"2c2aa48dbcf60bdbdb168697600e9f2adb33884ed73f2cf1f9f85d530a31e8dd"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1874,"index":26,"maximum_integer_bits":454,"maximum_rational_exponent_magnitude":57,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.312d000000000p-5"},"terminal_below_minimum":false,"transcript_bytes":4063,"witness_content_sha256":"fb22e23543ed0a63d0e55f16a97b2c01c866995cc3ed048c874d4c2471c5f5a0"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1867,"index":27,"maximum_integer_bits":450,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.312d000000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4062,"witness_content_sha256":"10a1cccf0ccedd10b7254f5a7630f02f497f13cbd43e6b852639ee1368414759"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1878,"index":28,"maximum_integer_bits":449,"maximum_rational_exponent_magnitude":132,"normalized_error":{"float_hex":"0x1.2376f08f7fb9dp-43"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.312d000000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7302,"witness_content_sha256":"71ea9b6b47215cfba2b7837893d7d6716cd439d2973b536c1fd102b98202b1a1"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1834,"index":29,"maximum_integer_bits":454,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.7d78400000000p-5"},"terminal_below_minimum":false,"transcript_bytes":4066,"witness_content_sha256":"c639d8ccc7d3b0f1106d7d3c40b01e77356ed30f5cf9e4b88294bcf244f1add4"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1828,"index":30,"maximum_integer_bits":450,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.7d78400000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4065,"witness_content_sha256":"84f9fe4d78a25e817f44253f261e1691b72a0cd1386ebc6b0fa85e8550f6efe2"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3771,"exact_operations":6264,"gcd_iterations":1837,"index":31,"maximum_integer_bits":447,"maximum_rational_exponent_magnitude":132,"normalized_error":{"float_hex":"0x1.6c2732e466f96p-43"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.7d78400000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7297,"witness_content_sha256":"0e5e5a19f94f136a9d30a51f0b307d3e24220c171fdd5009c4164f79b634ec53"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1810,"index":32,"maximum_integer_bits":452,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.3505b00000000p-5"},"terminal_below_minimum":false,"transcript_bytes":4078,"witness_content_sha256":"6da59f6a06eae60f937df8349c0d4037482c2bba9feec113281c8c1914c15356"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1788,"index":33,"maximum_integer_bits":448,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.3505b00000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4076,"witness_content_sha256":"6fa99abf144495d8fe5b5e83076fb689533477e9c33e77d4efa61ea6b6970ddf"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1779,"index":34,"maximum_integer_bits":447,"maximum_rational_exponent_magnitude":132,"normalized_error":{"float_hex":"0x1.707481a87dbd4p-42"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.3505b00000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7308,"witness_content_sha256":"3293aae514e300a324c21b2b412e4d4c7a80272132057dc8c59abd17c6946aac"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1854,"index":35,"maximum_integer_bits":455,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.cf88880000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4077,"witness_content_sha256":"649b19a6d5760d8967dce68cec062ae0d0ba68c8ecede30711cc7d1d7a036d09"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1841,"index":36,"maximum_integer_bits":451,"maximum_rational_exponent_magnitude":59,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.cf88880000000p-7"},"terminal_below_minimum":false,"transcript_bytes":4075,"witness_content_sha256":"a23e52b172d1b84da9a84a783815dd8e4c3600c767c9b794a2bd3707104e9405"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1809,"index":37,"maximum_integer_bits":451,"maximum_rational_exponent_magnitude":132,"normalized_error":{"float_hex":"0x1.ba468fbbc53c4p-44"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.cf88880000000p-8"},"terminal_below_minimum":false,"transcript_bytes":7310,"witness_content_sha256":"e5253e08d4ba510b9857fa90245d78f6d0c33863e4de24490b43bfd422176454"}}\x00\x00\x00\x00\x00\x00\x029{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":0,"certificate_passed":false,"disposition":"CERTIFICATE_REJECTION","dyadic_operations":476,"exact_operations":2969,"gcd_iterations":1896,"index":38,"maximum_integer_bits":453,"maximum_rational_exponent_magnitude":58,"normalized_error":null,"rational_operations":2493,"signed_step":{"float_hex":"0x1.5ba6660000000p-6"},"terminal_below_minimum":false,"transcript_bytes":4083,"witness_content_sha256":"71b4f091da9d9575a8e6f3393f70e9c5e7ff2a754e4eb4631f024c7c2ff536f9"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3763,"exact_operations":6256,"gcd_iterations":1891,"index":39,"maximum_integer_bits":451,"maximum_rational_exponent_magnitude":132,"normalized_error":{"float_hex":"0x1.4bf315b8fa7e5p-42"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.5ba6660000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7313,"witness_content_sha256":"b653c48ab19fbf5f7d12813c5c922db0a9a29281d47e74e30ff29c54fffcef9a"}}\x00\x00\x00\x00\x00\x00\x02O{"dataclass":"jxplanetx.engine.encounter.EncounterProposalRecord","fields":{"actual_force_evaluations":13,"certificate_passed":true,"disposition":"ACCEPTED","dyadic_operations":3758,"exact_operations":6251,"gcd_iterations":1871,"index":40,"maximum_integer_bits":449,"maximum_rational_exponent_magnitude":132,"normalized_error":{"float_hex":"0x1.4bde5cc6a35dep-42"},"rational_operations":2493,"signed_step":{"float_hex":"0x1.5ba6660000000p-7"},"terminal_below_minimum":false,"transcript_bytes":7319,"witness_content_sha256":"762ee3a2894a1d7b65ed2efadbba08680a761b3dc9afe05928b82ae6100b22d1"}}', 23495, 'c30405f0f3bce64c3b358a0d37706dde57b96ef8c883d6aa4c96081fbc8b414f'),
    'force_ledger_content_sha256': (b'jxplanetx.hybrid-private-encounter-force-ledger.content-integrity.v1\x00jxplanetx.hybrid-private-encounter-force-ledger.sequence.v1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x05\xc0{"dataclass":"jxplanetx.engine.evaluator.ForceLedgerEntry","fields":{"assumptions":["direct unsoftened point masses","finite-radius contact and singular coincidence are errors","repeatability is limited to the same backend, device, software stack, force order, body order, and tile size; no cross-device bitwise guarantee"],"backend_spec":{"dataclass":"jxplanetx.engine.contracts.BackendSpec","fields":{"allow_fallback":false,"backend_id":"numpy","determinism_scope":"SAME_RUNTIME_DEVICE","deterministic_reductions":true,"device":"cpu","dtype":"float64","fast_math":false,"tile_size":2}},"determinism_scope":"SAME_RUNTIME_DEVICE","evidence_class":"MODEL_OUTPUT","model_id":"force.newtonian.point_mass","order":0,"qualification_authorized":false,"registry_authorized":false,"role":"NEWTONIAN_BASE","source_ids":["STAR","PLANET"],"state_metadata":{"dataclass":"jxplanetx.engine.evaluator.StateMetadataBinding","fields":{"axes":"CARTESIAN_RIGHT_HANDED","body_ids":["STAR","PLANET"],"epoch":{"float_hex":"0x0.0p+0"},"frame":"BARYCENTRIC_INERTIAL","length_unit":"L","mass_unit":"M","origin":"BARYCENTER","provenance_citation":"Synthetic Wisdom--Holman fixture","provenance_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","provenance_source_id":"fixture.wh","provenance_version":"1","snapshot_id":"fixture.wh.circular-planet","time_scale":"SYNTHETIC","time_unit":"T","unit_system_id":"fixture.wh.units"}},"target_ids":["STAR","PLANET"],"tile_size":2}}', 1617, 'db73abb2022b465e7b63f828e6f183cc5a9dbfdb36fc2004f11f742639e7cab5'),
    'primary_counts_content_sha256': (b'jxplanetx.hybrid-private-encounter-primary-counts.content-integrity.v1\x00{"counts":{"dataclass":"jxplanetx.engine.encounter.EncounterExecutionCounts","fields":{"accepted_substeps":13,"candidate_domain_rejections":0,"certificate_rejections":27,"completed_rk_attempts":13,"derivative_domain_aborts":0,"error_rejections":0,"force_evaluations":169,"initialization_dyadic_operations":400,"initialization_exact_operations":921,"initialization_gcd_iterations":80,"initialization_rational_operations":521,"initialization_transcript_bytes":1863,"proposal_dyadic_operations":61693,"proposal_exact_operations":158858,"proposal_gcd_iterations":67109,"proposal_rational_operations":97165,"proposal_transcript_bytes":203909,"rejected_substeps":27,"stage_guard_aborts":0,"substep_proposals":40,"total_dyadic_operations":62093,"total_exact_operations":159779,"total_gcd_iterations":67189,"total_rational_operations":97686,"total_transcript_bytes":205772}},"schema":"jxplanetx.hybrid-private-encounter-primary-counts.payload.v1"}', 1010, 'bf733a488083fd9f966b1e1a977f2ac437439f2ab92d904a9686f6a8e193f8d3'),
    'accepted_steps_content_sha256': (b'jxplanetx.hybrid-private-encounter-accepted-steps.content-integrity.v1\x00jxplanetx.hybrid-private-encounter-accepted-steps.sequence.v1\x00\x00\x00\x00\x00\x00\x00\x00\r\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.0000000000000p-7"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.4000000000000p-7"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.9000000000000p-7"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.f400000000000p-8"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.3880000000000p-7"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.86a0000000000p-7"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.e848000000000p-8"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.312d000000000p-7"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.7d78400000000p-7"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.3505b00000000p-7"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.cf88880000000p-8"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.5ba6660000000p-7"}\x00\x00\x00\x00\x00\x00\x00${"float_hex":"0x1.5ba6660000000p-7"}', 713, '8c9bee865756515a47dec1ad776f3c1fcd25bff2a7c61f206ec65a0fa4211631'),
    'private_execution_content_sha256': (b'jxplanetx.hybrid-private-encounter-execution.content-integrity.v1\x00{"component_digests":{"accepted_steps_content_sha256":"8c9bee865756515a47dec1ad776f3c1fcd25bff2a7c61f206ec65a0fa4211631","final_state_content_sha256":"816c7feb2cad928c1609a2c7a6cdd9cde1eb377347b01b80391e0be8da72c4e4","force_ledger_content_sha256":"db73abb2022b465e7b63f828e6f183cc5a9dbfdb36fc2004f11f742639e7cab5","initialization_content_sha256":"fc818f80229f200f1d6db6d03f3020b3cf6e3336fff9536c0d90a82e65a5b66e","primary_counts_content_sha256":"bf733a488083fd9f966b1e1a977f2ac437439f2ab92d904a9686f6a8e193f8d3","proposal_ledger_content_sha256":"c30405f0f3bce64c3b358a0d37706dde57b96ef8c883d6aa4c96081fbc8b414f"},"method_id":"integrator.adaptive.rkf78.encounter_segment_newtonian_v1","outer_step_index":1,"schema":"jxplanetx.hybrid-private-encounter-execution-manifest.payload.v1","segment_spec":{"dataclass":"jxplanetx.engine.encounter_contracts.AdaptiveEncounterSegmentSpec","fields":{"acceleration_bound_policy":"under the simultaneous pre-crossing hypothesis d_ik(t)>rho_ik, set A_i=sum_{k!=i}(GM_k/rho_ik^2) in immutable k order and A_ij=A_i+A_j; all operands and sums are custom exact rationals","acceptance_policy":"after a completed finite candidate-domain-safe 13-stage attempt accept iff the normalized maximum pair-and-centroid error is <=1; error=0 uses maximum_scale_factor,otherwise clamp safety_factor*error^(-1/8) between minimum_scale_factor and maximum_scale_factor","accepted_order":8,"accepted_solution":"ORDER_8_HATTED","accepted_state_accumulation":"KAHAN_NUMPY_FLOAT64_COMPONENTWISE_POSITIONS_AND_VELOCITIES","adaptive":true,"allowed_axes":["ICRS_ALIGNED","CARTESIAN_RIGHT_HANDED"],"allowed_time_scales":["TDB","SYNTHETIC"],"backend_scope":"EXACT_NUMPY_CPU_FLOAT64_ONLY","body_order":["STAR","PLANET"],"body_order_policy":"IMMUTABLE_STATE_BODY_ORDER_WITH_CANONICAL_PAIRS_I_LESS_THAN_J","carry_entry_policy":"initialize position_carry and velocity_carry as owned exact positive-zero NumPy float64 arrays at the start of every standalone encounter segment and every hybrid full-macrostep replacement; never import a carry from a discarded Wisdom-Holman candidate or a preceding outer segment","certificate_cadence_policy":"evaluate one all-pair simultaneous certificate from the currently accepted state before any RK stage force call for every proposed substep; a false strict inequality is a recoverable certificate rejection reduced by the fixed domain-rejection factor unless a step/rejection cap then requires fatal fail-closed termination; an uncertified proposal is not evidence that a collision or floor crossing occurs","certificate_claim_scope":"EXACT_NEWTONIAN_LOCAL_IVP_ISSUING_FROM_EACH_ACCEPTED_NUMERICAL_NODE_IS_NONCOLLIDING_WITH_RESPECT_TO_RETAINED_PAIR_FLOORS_ON_THAT_SUBSTEP_ONLY","certificate_nonclaims":"NO_EVENT_TIME_OR_MINIMUM_DISTANCE_LOCALIZATION;NO_COLLISION_RESPONSE;NO_CLEARANCE_BEFORE_THE_INITIAL_STATE_OR_AFTER_THE_SEGMENT_ENDPOINT;NO_GLOBAL_OR_FUTURE_CLEARANCE;NO_TRAJECTORY_ACCURACY_FROM_THE_CLEARANCE_CERTIFICATE;NO_EXISTENCE_CLAIM_BEYOND_THE_ACCEPTED_LOCAL_IVP","checksum_binding_policy":"pair-table checksum binds body order,floors,pair position/velocity atols and rtols,and GM-centroid tolerances; schedule and result checksums bind initial_epoch,the separate explicit endpoint_epoch,the retained signed duration,all accepted signed substeps,proposal dispositions,actual force counts,initialization and proposal witness digests,resource specifications,input provenance,primary accounting,and replay accounting","collision_response":false,"completed_trial_array_policy":"after thirteen finite derivatives require accepted_positions,accepted_velocities,embedded_positions,embedded_velocities,position_defect,velocity_defect,accepted_position_carry,and accepted_velocity_carry all be exact NumPy float64 arrays of the state shape and entirely finite before candidate-domain or normalized-error evaluation","controller_exponent":{"float_hex":"0x1.0000000000000p-3"},"defect_orientation":"ACCEPTED_ORDER_8_MINUS_EMBEDDED_ORDER_7","dense_output":false,"domain_rejection_scale_factor":{"float_hex":"0x1.0000000000000p-1"},"dtype":"float64","duration":{"float_hex":"0x1.0000000000000p-3"},"effective_floor_policy":"rho_ij is exactly the retained canonical pair_certification_floors entry; runtime preflight converts rho_ij and both retained radii to canonical exact dyadics and requires rho_ij>=radius_i+radius_j before any force call","embedded_order":7,"embedded_solution":"ORDER_7_ORDINARY","endpoint_epoch":{"float_hex":"0x1.0000000000000p-3"},"error_norm":"NORMALIZED_MAX_OF_ALL_CANONICAL_PAIR_RELATIVE_COMPONENTS_AND_GM_CENTROID_COMPONENTS","event_detection":false,"evidence_class":"MODEL_OUTPUT","exact_rational_resources":{"dataclass":"jxplanetx.engine.encounter_contracts.EncounterExactRationalResourceSpec","fields":{"exponent_policy":"for nonzero p/q define e=floor(log2(abs(p/q))) by integer bit lengths plus one exact shifted-integer comparison; define e=0 for zero; require abs(e) no greater than maximum_rational_exponent_magnitude; for every retained nonzero canonical dyadic (m,e),the diagnostic observes both abs(e) and abs(e+bit_length(abs(m))-1),the latter being the value floor-log2,and zero contributes zero; maximum_rational_exponent_magnitude in each usage record is the maximum of those general-rational and canonical-dyadic observations only and deliberately excludes a larger pre-normalization constructor exponent,alignment exponent or shift count,and any intermediate that canonicalizes or cancels to zero,although all such paths remain prospectively subject to the fixed exponent and integer-bit caps","maximum_body_count":16,"maximum_gcd_iterations_per_proposal":500000,"maximum_gcd_iterations_per_reduction":16384,"maximum_gcd_iterations_per_segment":64500000,"maximum_initialization_gcd_iterations":500000,"maximum_initialization_operations":100000,"maximum_initialization_transcript_bytes":1048576,"maximum_integer_bits":8192,"maximum_operations_per_proposal":650000,"maximum_operations_per_segment":83450000,"maximum_pair_count":120,"maximum_rational_exponent_magnitude":4096,"maximum_witness_diagnostic_bytes_per_proposal":4096,"maximum_witness_ledger_bytes":16777216,"maximum_witness_transcript_bytes_per_proposal":1310720,"maximum_witness_transcript_bytes_per_segment":168820736,"no_floating_fallback":true,"operation_policy":"general p/q primitives are FROM_BINARY64,ADD,SUBTRACT,MULTIPLY,DIVIDE,SQUARE,NEGATE,and COMPARE with sign normalization and Euclidean reduction after construction; dyadic primitives are FROM_BINARY64,ADD,SUBTRACT,MULTIPLY,SQUARE,NEGATE,and COMPARE,align ADD/SUBTRACT to the smaller exponent,and canonicalize by shifting every factor of two from m into e; for each dyadic distance guard convert the 3N coordinates once,form each component difference,and accumulate (dx^2+dy^2)+dz^2 left-to-right before one strict comparison with the cached dyadic rho^2; fixed vector-component and canonical-pair order is mandatory in both lanes","representation":"two disjoint custom exact lanes: general values used by the simultaneous first-crossing certificate are canonical built-in-integer pairs (p,q) with q>0,gcd(abs(p),q)=1,and zero (0,1); binary64 duration scheduling and all radius-sum,start,stage,and candidate squared-distance guards use canonical exact dyadics (m,e) denoting m*2^e,with zero (0,0) and every nonzero m odd; finite binary64 enters either lane only through a prospectively authorized float.as_integer_ratio conversion; Fraction,Decimal,third-party exact code,and floating comparison fallback are forbidden","resource_policy":"report general-rational and dyadic deterministic abstract exact-work units separately and their exact sum; the fixed named path weights are exactly ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHTS_V1 and are retained verbatim in the resource specification; a selected named path charges its table weight once at its explicit runtime charge site,including conditional nonzero,even-normalization,nextafter,and conversion paths,while unselected conditional paths charge zero; these are qualification weights for this algorithm,not counts of Python,C,CPU,big-integer,or machine instructions or primitives and not estimates or guarantees of elapsed time,wall time,or memory; each charge site prospectively authorizes its complete table weight against the initialization-plus-segment ledgers during preflight or proposal-plus-segment ledgers during a proposal before the selected path produces its result; explicit integer-width and exponent checks additionally authorize potentially enlarged shift,add,subtract,multiply,and conversion results before those result integers are formed; reduction uses a custom interruptible Euclidean loop only in the general p/q lane and charges exactly one separately reported GCD.EUCLIDEAN_ITERATION unit per selected loop iteration,without claiming a count of underlying modulus or integer primitives; this unit is charged to separate initialization/proposal and segment gcd ledgers without double-charging exact-work units; it also obeys the per-reduction iteration cap; body,pair,and identifier caps are checked before numeric conversion or set allocation,and full initialization and per-proposal operation reservations are required from the remaining segment budget; every normalized rational exponent and integer intermediate is capped; any failed prospective check is fatal and never falls back to opaque math.gcd or floating arithmetic; the dyadic lane performs no GCD","work_weight_table":[["GENERAL_RATIONAL","INTEGER_WIDTH_OBSERVATION",2],["GENERAL_RATIONAL","RATIONAL_EXPONENT_DIAGNOSTIC",1],["GENERAL_RATIONAL","INTEGER_COMPARE",1],["GENERAL_RATIONAL","INTEGER_ABSOLUTE",1],["GENERAL_RATIONAL","INTEGER_NEGATE",1],["GENERAL_RATIONAL","INTEGER_SHIFT_LEFT",1],["GENERAL_RATIONAL","INTEGER_ADD",1],["GENERAL_RATIONAL","INTEGER_SUBTRACT",1],["GENERAL_RATIONAL","INTEGER_MULTIPLY",1],["GENERAL_RATIONAL","INTEGER_FLOOR_DIVIDE",1],["GENERAL_RATIONAL","BINARY64_ENVELOPE_BASE",8],["GENERAL_RATIONAL","BINARY64_ENVELOPE_NONZERO",4],["GENERAL_RATIONAL","RATIONAL_CONSTRUCTION",1],["GENERAL_RATIONAL","FROM_BINARY64_RATIO_EXTRACTION",1],["GENERAL_RATIONAL","SIGNED_STEP_ABSOLUTE",1],["DYADIC","INTEGER_WIDTH_OBSERVATION",2],["DYADIC","BINARY64_ENVELOPE_BASE",8],["DYADIC","BINARY64_ENVELOPE_NONZERO",4],["DYADIC","DYADIC_CONSTRUCTION",1],["DYADIC","INPUT_EXPONENT_CHECK",1],["DYADIC","NONZERO_ODDNESS_BRANCH",1],["DYADIC","EVEN_LOWBIT_PATH",3],["DYADIC","EVEN_CANONICALIZE_PATH",3],["DYADIC","CANONICAL_EXPONENT_DIAGNOSTIC",4],["DYADIC","FROM_BINARY64_RATIO_EXTRACTION",1],["DYADIC","FROM_BINARY64_POWER_OF_TWO_CHECK",2],["DYADIC","ADD_OR_SUBTRACT_ALIGN",3],["DYADIC","ADD_OR_SUBTRACT_SHIFT_PAIR",2],["DYADIC","ADD_OR_SUBTRACT_COMBINE",1],["DYADIC","MULTIPLY_MANTISSA_AND_EXPONENT",2],["DYADIC","NEGATE",1],["DYADIC","COMPARE_ALIGN_AND_SHIFT_PAIR",5],["DYADIC","COMPARE_RESULT",1],["DYADIC","TO_BINARY64_RATIO_BUILD",1],["DYADIC","TO_BINARY64_ROUND",1],["DYADIC","DURATION_ABSOLUTE",1],["DYADIC","SCHEDULER_NEXTAFTER_DOWN",1],["DYADIC","SCHEDULER_SUCCESSOR_PROBE",1],["GCD","EUCLIDEAN_ITERATION",1]],"work_weight_table_id":"ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHTS_V1"}},"exact_witness_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_JSON_V1","exact_witness_checksum_domain":"jxplanetx.encounter-exact-clearance-witness.content-integrity.v1","exactly_time_reversible":false,"first_crossing_certificate":"for every canonical pair simultaneously require the strict exact-rational inequality ell_ij^2 > (rho_ij+(1/2)*(A_i+A_j)*Delta^2)^2; if an earliest first crossing existed, all pair floors would hold beforehand, the acceleration bounds would bound the relative Taylor remainder by (1/2)*(A_i+A_j)*u^2, and the strict inequality would contradict contact at that crossing; equality is uncertified and is rejected","force_evaluation_accounting":"record per proposal its disposition and actual derivative/force calls; certificate rejection has zero; stage-guard abort has zero through twelve; derivative-domain abort has one through thirteen; completed RK attempt has exactly thirteen finite derivative calls; total force evaluations equal the exact sum of per-proposal calls,not thirteen times all proposals","force_plan_scope":"AUTONOMOUS_MUTUAL_ALL_ACTIVE_POSITIVE_GM_UNSOFTENED_NEWTONIAN_ONLY","globally_symplectic":false,"gm_centroid_error_scale":"C_x=sum_i(GM_i*x_i)/sum_i(GM_i) and C_v analogously in immutable body order; centroid defect is sum_i(GM_i*trial.position_defect_i)/sum_i(GM_i) or the velocity-defect analogue directly from the frozen tableau and scale is centroid_atol+centroid_rtol*max(abs(current_centroid_component),abs(accepted_centroid_component)); position and velocity controls are separate and all reductions use fixed left-to-right NumPy float64 order","gm_centroid_position_atol":{"float_hex":"0x1.0c6f7a0b5ed8dp-20"},"gm_centroid_position_rtol":{"float_hex":"0x1.19799812dea11p-40"},"gm_centroid_velocity_atol":{"float_hex":"0x1.0c6f7a0b5ed8dp-20"},"gm_centroid_velocity_rtol":{"float_hex":"0x1.19799812dea11p-40"},"initial_domain_policy":"before any force call require the accepted initial numerical node finite and,for every canonical pair,the exact-dyadic squared separation strictly greater than rho_ij^2; failure is fatal and returns no result","initial_epoch":{"float_hex":"0x0.0p+0"},"initial_step_magnitude":{"float_hex":"0x1.0000000000000p-3"},"initialization_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_JSON_V1","initialization_checksum_domain":"jxplanetx.encounter-initialization.content-integrity.v1","initialization_resource_policy":"before proposal one,charge every exact GM,radius,floor,and initial-position conversion,every rho_ij>=radius_i+radius_j proof,every initial squared-separation>rho_ij^2 proof,and construction of all cached A_i=sum_{k!=i}(GM_k/rho_ik^2) values to a separately capped initialization dyadic-work,general-rational-work,gcd,transcript ledger and also to the enclosing lane\'s whole-segment ledgers; retain a domain-separated initialization transcript digest plus diagnostics bounded by maximum_witness_diagnostic_bytes_per_proposal; successful work is never hidden in a segment-only pool or charged to proposal one,while any initialization failure is fatal and releases no partial result; mandatory replay recomputes initialization under an identical separate ledger","input_binding_policy":"snapshot body_ids must exactly equal body_order,snapshot.epoch must be an exact built-in float bit-identical to initial_epoch,and metadata must name the required frame,origin,allowed axes,and allowed time scale; positions,velocities,GM,masses,and radii must be exact CPU numpy.ndarray objects with float64 dtype,required shapes,C-contiguous finite storage,and no subclass or device transfer; massive must be an exact CPU NumPy boolean array containing only true; every GM is strictly positive,every radius is nonnegative,and physical masses are retained but unused by the GM-scaled dynamics; the force plan is exactly one NewtonianPointMass with source_ids=target_ids=body_order on numpy.cpu and contains no other force; all parameter-metadata validity intervals must cover the closed public label interval between initial_epoch and endpoint_epoch","linear_minimum_policy":"for one proposed signed step h_s set Delta=abs(h_s), r_ij=x_j-x_i, w_ij=sign(h_s)*(v_j-v_i), b=dot(r_ij,w_ij), c=dot(w_ij,w_ij),and q=dot(r_ij,r_ij), all exactly; the exact minimum ell^2 of norm(r_ij+u*w_ij)^2 on 0<=u<=Delta is q when c=0 or b>=0, norm(r_ij+Delta*w_ij)^2 when b<0 and -b>=c*Delta, and q-b^2/c otherwise; branch comparisons and ell^2 are exact rational","maximum_accepted_substeps":128,"maximum_consecutive_rejections":32,"maximum_force_evaluations":1664,"maximum_rejected_substeps":64,"maximum_scale_factor":{"float_hex":"0x1.4000000000000p+2"},"maximum_step_magnitude":{"float_hex":"0x1.0000000000000p-3"},"maximum_substep_proposals":128,"method_class":"EXPLICIT_EMBEDDED_RUNGE_KUTTA_LOCAL_ENCOUNTER_IVP","method_id":"integrator.adaptive.rkf78.encounter_segment_newtonian_v1","minimum_scale_factor":{"float_hex":"0x1.999999999999ap-3"},"minimum_step_failure_policy":"a proposal rejected at minimum_step_magnitude,an exact remaining chunk below that magnitude except a deterministic endpoint-remainder chunk,or a rejected terminal endpoint-remainder chunk strictly below minimum-step magnitude,or any exhausted proposal,acceptance,rejection,force,or exact-resource cap is fatal; an ordinary clipped remainder strictly greater than minimum may reject and retry or split,while equality is fatal on rejection","minimum_step_magnitude":{"float_hex":"0x1.0624dd2f1a9fcp-10"},"pair_certification_floors":[{"float_hex":"0x1.47ae147ae147bp-7"}],"pair_error_scale":"for pair i<j and component k, defect is (trial.position_defect_j-trial.position_defect_i)_k or (trial.velocity_defect_j-trial.velocity_defect_i)_k directly from the frozen Fehlberg tableau result,never recomputed from Kahan-updated accepted minus embedded states; scale is pair_atol_ij+pair_rtol*max(abs(current_j-current_i)_k,abs(accepted_j-accepted_i)_k); apply separately to position and velocity","pair_position_atols":[{"float_hex":"0x1.0c6f7a0b5ed8dp-20"}],"pair_position_rtol":{"float_hex":"0x1.19799812dea11p-40"},"pair_table_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1","pair_table_checksum_domain":"jxplanetx.encounter-pair-table.content-integrity.v1","pair_table_policy":"for N bodies the canonical table has exactly N*(N-1)/2 entries ordered (0,1),(0,2),...,(0,N-1),(1,2),...,(N-2,N-1); each supplied binary64 certification floor is positive, retained verbatim, and must be exact-dyadically greater than or equal to the exact sum of the two retained nonnegative binary64 radii; no scalar or hidden default floor","pair_velocity_atols":[{"float_hex":"0x1.0c6f7a0b5ed8dp-20"}],"pair_velocity_rtol":{"float_hex":"0x1.19799812dea11p-40"},"principal_order":8,"proposal_accounting":"proposals=certificate_rejections+stage_guard_aborts+derivative_domain_aborts+completed_rk_attempts; completed_rk_attempts=candidate_domain_rejections+error_rejections+accepted_substeps; rejected_substeps=certificate_rejections+stage_guard_aborts+derivative_domain_aborts+candidate_domain_rejections+error_rejections; total and consecutive rejection caps apply to this exact five-category sum; all counts are exact nonnegative built-in integers","public_execution_accounting_scope":"PRIMARY_PLUS_MANDATORY_SEMANTIC_REPLAY_WITH_SEPARATE_AND_TOTAL_COUNTERS","qualification_authorized":false,"registry_authorized":false,"required_frame":"BARYCENTRIC_INERTIAL","required_origin":"BARYCENTER","resource_lane_policy":"primary execution and mandatory semantic replay each receive separate identical proposal,acceptance,rejection,force,body,pair,integer,exponent,gcd,operation,transcript,and witness caps,including a separate initialization ledger in each lane; report initialization,proposal,lane-total,primary,replay,and public-total usage separately; the public-total force,exact-operation,GCD,and transcript ceilings are exactly twice the corresponding one-lane ceilings; neither lane may borrow unused budget from the other","resource_nonclaim_policy":"fixed logical caps make arithmetic and custody finite but are not a wall-clock or resident-memory denial-of-service guarantee; an external service must additionally enforce process isolation,time,and memory limits","result_content_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1","result_content_checksum_domain":"jxplanetx.encounter-segment-result.content-integrity.v1","result_custody_policy":"result arrays and ledgers are owned immutable copies bound to input provenance,body order,pair tables,tolerances,resources,duration,schedule,primary accounting,and replay accounting by domain-separated checksums","safety_factor":{"float_hex":"0x1.ccccccccccccdp-1"},"schedule_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1","schedule_checksum_domain":"jxplanetx.encounter-segment-schedule.content-integrity.v1","source_author":"Erwin Fehlberg","source_document_id":"19680027281","source_publication_date":"1968-10-01","source_report":"NASA-TR-R-287","source_title":"Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta Formulas with Stepsize Control","source_url":"https://ntrs.nasa.gov/citations/19680027281","stage_abort_policy":"a failed pre-derivative stage guard aborts that RK attempt immediately and retains the actual number from zero through twelve of derivative/force calls already completed; a nonfinite derivative is a distinct derivative-domain abort retaining the actual number from one through thirteen of calls made; only thirteen finite derivatives constitute a completed RK attempt,and candidate-domain and error rejections exist only after such a completed attempt","stage_count":13,"stage_epoch_policy":"CONSTANT_INITIAL_PUBLIC_LABEL_FOR_ALL_AUTONOMOUS_NEWTONIAN_STAGE_EVALUATIONS; every derivative evaluation presents the bit-identical initial_epoch public label to force metadata and ignores the RK stage-time label; exact local offsets,accepted signed substeps,proposal signed steps,and the fixed RKF78 c-table identify mathematical stage times separately; the full closed public label interval between initial_epoch and endpoint_epoch remains subject to parameter-metadata preflight validation; no public label alters or supplies a retained RK substep or force","stage_guard_policy":"before each of the 13 derivative calls require every trial-stage position and velocity finite and exact-dyadic squared pair separation strictly greater than rho_ij^2; after a completed 13-stage attempt apply the same finite strict guard to the eighth-order candidate before error acceptance; a trial-local nonfinite stage,candidate,defect,derivative,or a stage/candidate at or below a floor is not an event and is a recoverable fixed-factor domain rejection; accepted-state nonfiniteness,initial floor failure,or exact-resource failure is fatal","step_scheduler_policy":"proposals have sign(h); set exact positive target to min(exact remaining duration,exact proposed positive binary64 magnitude,exact maximum-step magnitude); convert target once by correctly-rounded nearest-ties-to-even binary64,then if that value is greater than target by exact-dyadic comparison apply math.nextafter(value,0.0) once; require the result finite and positive,which is the unique largest positive binary64 no greater than target; exact-subtract each accepted magnitude from remaining and repeat until exact zero; every chunk remains subject to the certificate,stage guards,and error test; a clipped chunk strictly greater than minimum may reject and retry/split normally,while a terminal chunk <minimum is acceptance-only and any rejection is fatal; fail closed if conversion,progress,exact subtraction,or endpoint completion violates a step or resource cap","tableau_id":"NASA_TR_R_287_FEHLBERG_7_8_13_STAGE","time_policy":"retain initial_epoch,an externally supplied integrity-bound but unauthenticated provenance-only endpoint_epoch label,and one nonzero signed binary64 mathematical duration h; only finiteness and strict monotonicity from initial_epoch in sign(h) are internally checkable,so endpoint_epoch need not equal float(initial_epoch+h) and is not evidence of outer-lattice construction; advance state on an authoritative local offset lattice from exact zero to exact h and never recover any RK step or state duration from endpoint_epoch-initial_epoch; accepted signed binary64 substeps are exact-rationally accounted and their mathematical sum must equal h exactly","transaction_policy":"copy validated inputs to owned work buffers before evaluation; never mutate caller buffers; keep accepted state,carries,local offset,and exact duration ledger unchanged until certificate,all stages,candidate guard,and error acceptance succeed; discard every rejected trial completely; on any fatal failure return no partial result","validation_replay_count":1,"validation_replay_policy":"one full deterministic replay from owned input recomputes adaptive decisions,exact certificates,stage guards,states,carries,local offsets,proposal dispositions,actual force counts,and witness hashes; require bitwise-identical arrays and identical scalar ledgers before release; resource caps and usage obey the separate-lane resource policy","witness_custody_policy":"retain per proposal only canonical pair/disposition identifiers,actual force-call count,bounded binary64 diagnostics,bounded integer bit-length diagnostics,and a domain-separated SHA-256 of the canonical exact comparison transcript; never retain unbounded rational numerators or denominators; replay recomputes every exact inequality","witness_serialization_policy":"for initialization and each proposal serialize UTF-8 JSON with sort_keys=True,ensure_ascii=True,separators=(\',\',\':\'),allow_nan=False; payload contains method/version,phase,proposal index or null,signed-step float.hex or null,disposition,actual force calls,canonical pair indices,linear-minimum branch,and every exact comparison operand as either canonical base-10 numerator and positive denominator strings or canonical base-10 dyadic mantissa and built-in integer exponent; charge and prospectively cap the decimal and JSON byte envelope before conversion,serialization,retention,or hashing,and then charge every actual UTF-8 transcript byte before streaming domain+\'\\\\0\'+payload into SHA-256; retain only the digest and bounded diagnostics,not the serialized exact operands"}},"summary":{"accepted_substep_count":13,"dyadic_operation_count":62093,"force_evaluations":169,"gcd_iteration_count":67189,"general_rational_operation_count":97686,"maximum_integer_bits":465,"maximum_rational_exponent_magnitude":146,"proposal_count":40,"rejected_substep_count":27,"streamed_witness_ledger_bytes":23393,"transcript_byte_count":205772}}', 25881, '8a0235b549878a5cb75b22872d2c434c5f14b6f5f051abbcbc94ecc799f44789'),
}

_HYBRID_STEP_LEDGER_LITERAL_KAT = (b'jxplanetx.hybrid-wh-rkf78-step-ledger.content-integrity.v1\x00{"records":[{"dataclass":"jxplanetx.engine.hybrid.HybridOuterStepRecord","fields":{"decision":{"dataclass":"jxplanetx.engine.hybrid_contracts.HybridFarProbeDecision","fields":{"body_count":2,"body_indices":[],"outcome":"NEAR_SWITCH","pair_indices":[[0,1]],"phase":"POST_DRIFT_PRE_FORCE_PATH_SCREEN","reason":"FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN","work":{"dataclass":"jxplanetx.engine.hybrid_contracts.HybridFarProbeWork","fields":{"cartesian_to_jacobi_calls_entered":1,"cartesian_to_jacobi_transforms_completed":1,"center_of_mass_drifts_completed":1,"first_half_kicks_completed":1,"force_calls_entered":1,"force_evaluations_completed":1,"interaction_force_assemblies":1,"jacobi_to_cartesian_calls_entered":1,"jacobi_to_cartesian_transforms_completed":1,"kepler_probe_records":[{"dataclass":"jxplanetx.engine.hybrid_contracts.HybridKeplerProbeRecord","fields":{"bracket_expansions_entered":0,"call_completed":true,"completed_series_bundle_count":2,"g_bundle_calls_completed":2,"g_bundle_calls_entered":2,"interrupted_series_terms":0,"iterations_entered":1,"secondary_index":1,"source":"FROZEN_PRIVATE_WISDOM_HOLMAN_ELLIPTIC_UNIVERSAL_KEPLER_SUBFLOW","terminal_status":"COMPLETED"}}],"kepler_subflow_calls_entered":1,"kepler_subflow_solves_completed":1,"node_guard_evaluations":2,"path_guard_evaluations":1,"second_half_kicks_completed":0,"universal_g_bundle_calls_completed":2,"universal_g_bundle_calls_entered":2,"universal_series_terms_evaluated":512,"universal_solver_bracket_expansions":0,"universal_solver_iterations":1}}}},"endpoint_epoch":{"float_hex":"0x1.0000000000000p-3"},"mode":"CARTESIAN_RKF78_NEAR_FULL_INTERVAL","outer_step_index":1,"private_encounter":{"dataclass":"jxplanetx.engine.hybrid_contracts.HybridPrivateEncounterRecord","fields":{"accepted_signed_steps":[{"float_hex":"0x1.0000000000000p-7"},{"float_hex":"0x1.4000000000000p-7"},{"float_hex":"0x1.9000000000000p-7"},{"float_hex":"0x1.f400000000000p-8"},{"float_hex":"0x1.3880000000000p-7"},{"float_hex":"0x1.86a0000000000p-7"},{"float_hex":"0x1.e848000000000p-8"},{"float_hex":"0x1.312d000000000p-7"},{"float_hex":"0x1.7d78400000000p-7"},{"float_hex":"0x1.3505b00000000p-7"},{"float_hex":"0x1.cf88880000000p-8"},{"float_hex":"0x1.5ba6660000000p-7"},{"float_hex":"0x1.5ba6660000000p-7"}],"accepted_steps_content_sha256":"8c9bee865756515a47dec1ad776f3c1fcd25bff2a7c61f206ec65a0fa4211631","accepted_substep_count":13,"dyadic_operation_count":62093,"final_state_content_sha256":"816c7feb2cad928c1609a2c7a6cdd9cde1eb377347b01b80391e0be8da72c4e4","force_evaluations":169,"force_ledger_content_sha256":"db73abb2022b465e7b63f828e6f183cc5a9dbfdb36fc2004f11f742639e7cab5","gcd_iteration_count":67189,"general_rational_operation_count":97686,"initialization_content_sha256":"fc818f80229f200f1d6db6d03f3020b3cf6e3336fff9536c0d90a82e65a5b66e","maximum_integer_bits":465,"maximum_rational_exponent_magnitude":146,"outer_step_index":1,"primary_counts_content_sha256":"bf733a488083fd9f966b1e1a977f2ac437439f2ab92d904a9686f6a8e193f8d3","private_execution_content_sha256":"8a0235b549878a5cb75b22872d2c434c5f14b6f5f051abbcbc94ecc799f44789","proposal_count":40,"proposal_ledger_content_sha256":"c30405f0f3bce64c3b358a0d37706dde57b96ef8c883d6aa4c96081fbc8b414f","rejected_substep_count":27,"segment_spec":{"dataclass":"jxplanetx.engine.encounter_contracts.AdaptiveEncounterSegmentSpec","fields":{"acceleration_bound_policy":"under the simultaneous pre-crossing hypothesis d_ik(t)>rho_ik, set A_i=sum_{k!=i}(GM_k/rho_ik^2) in immutable k order and A_ij=A_i+A_j; all operands and sums are custom exact rationals","acceptance_policy":"after a completed finite candidate-domain-safe 13-stage attempt accept iff the normalized maximum pair-and-centroid error is <=1; error=0 uses maximum_scale_factor,otherwise clamp safety_factor*error^(-1/8) between minimum_scale_factor and maximum_scale_factor","accepted_order":8,"accepted_solution":"ORDER_8_HATTED","accepted_state_accumulation":"KAHAN_NUMPY_FLOAT64_COMPONENTWISE_POSITIONS_AND_VELOCITIES","adaptive":true,"allowed_axes":["ICRS_ALIGNED","CARTESIAN_RIGHT_HANDED"],"allowed_time_scales":["TDB","SYNTHETIC"],"backend_scope":"EXACT_NUMPY_CPU_FLOAT64_ONLY","body_order":["STAR","PLANET"],"body_order_policy":"IMMUTABLE_STATE_BODY_ORDER_WITH_CANONICAL_PAIRS_I_LESS_THAN_J","carry_entry_policy":"initialize position_carry and velocity_carry as owned exact positive-zero NumPy float64 arrays at the start of every standalone encounter segment and every hybrid full-macrostep replacement; never import a carry from a discarded Wisdom-Holman candidate or a preceding outer segment","certificate_cadence_policy":"evaluate one all-pair simultaneous certificate from the currently accepted state before any RK stage force call for every proposed substep; a false strict inequality is a recoverable certificate rejection reduced by the fixed domain-rejection factor unless a step/rejection cap then requires fatal fail-closed termination; an uncertified proposal is not evidence that a collision or floor crossing occurs","certificate_claim_scope":"EXACT_NEWTONIAN_LOCAL_IVP_ISSUING_FROM_EACH_ACCEPTED_NUMERICAL_NODE_IS_NONCOLLIDING_WITH_RESPECT_TO_RETAINED_PAIR_FLOORS_ON_THAT_SUBSTEP_ONLY","certificate_nonclaims":"NO_EVENT_TIME_OR_MINIMUM_DISTANCE_LOCALIZATION;NO_COLLISION_RESPONSE;NO_CLEARANCE_BEFORE_THE_INITIAL_STATE_OR_AFTER_THE_SEGMENT_ENDPOINT;NO_GLOBAL_OR_FUTURE_CLEARANCE;NO_TRAJECTORY_ACCURACY_FROM_THE_CLEARANCE_CERTIFICATE;NO_EXISTENCE_CLAIM_BEYOND_THE_ACCEPTED_LOCAL_IVP","checksum_binding_policy":"pair-table checksum binds body order,floors,pair position/velocity atols and rtols,and GM-centroid tolerances; schedule and result checksums bind initial_epoch,the separate explicit endpoint_epoch,the retained signed duration,all accepted signed substeps,proposal dispositions,actual force counts,initialization and proposal witness digests,resource specifications,input provenance,primary accounting,and replay accounting","collision_response":false,"completed_trial_array_policy":"after thirteen finite derivatives require accepted_positions,accepted_velocities,embedded_positions,embedded_velocities,position_defect,velocity_defect,accepted_position_carry,and accepted_velocity_carry all be exact NumPy float64 arrays of the state shape and entirely finite before candidate-domain or normalized-error evaluation","controller_exponent":{"float_hex":"0x1.0000000000000p-3"},"defect_orientation":"ACCEPTED_ORDER_8_MINUS_EMBEDDED_ORDER_7","dense_output":false,"domain_rejection_scale_factor":{"float_hex":"0x1.0000000000000p-1"},"dtype":"float64","duration":{"float_hex":"0x1.0000000000000p-3"},"effective_floor_policy":"rho_ij is exactly the retained canonical pair_certification_floors entry; runtime preflight converts rho_ij and both retained radii to canonical exact dyadics and requires rho_ij>=radius_i+radius_j before any force call","embedded_order":7,"embedded_solution":"ORDER_7_ORDINARY","endpoint_epoch":{"float_hex":"0x1.0000000000000p-3"},"error_norm":"NORMALIZED_MAX_OF_ALL_CANONICAL_PAIR_RELATIVE_COMPONENTS_AND_GM_CENTROID_COMPONENTS","event_detection":false,"evidence_class":"MODEL_OUTPUT","exact_rational_resources":{"dataclass":"jxplanetx.engine.encounter_contracts.EncounterExactRationalResourceSpec","fields":{"exponent_policy":"for nonzero p/q define e=floor(log2(abs(p/q))) by integer bit lengths plus one exact shifted-integer comparison; define e=0 for zero; require abs(e) no greater than maximum_rational_exponent_magnitude; for every retained nonzero canonical dyadic (m,e),the diagnostic observes both abs(e) and abs(e+bit_length(abs(m))-1),the latter being the value floor-log2,and zero contributes zero; maximum_rational_exponent_magnitude in each usage record is the maximum of those general-rational and canonical-dyadic observations only and deliberately excludes a larger pre-normalization constructor exponent,alignment exponent or shift count,and any intermediate that canonicalizes or cancels to zero,although all such paths remain prospectively subject to the fixed exponent and integer-bit caps","maximum_body_count":16,"maximum_gcd_iterations_per_proposal":500000,"maximum_gcd_iterations_per_reduction":16384,"maximum_gcd_iterations_per_segment":64500000,"maximum_initialization_gcd_iterations":500000,"maximum_initialization_operations":100000,"maximum_initialization_transcript_bytes":1048576,"maximum_integer_bits":8192,"maximum_operations_per_proposal":650000,"maximum_operations_per_segment":83450000,"maximum_pair_count":120,"maximum_rational_exponent_magnitude":4096,"maximum_witness_diagnostic_bytes_per_proposal":4096,"maximum_witness_ledger_bytes":16777216,"maximum_witness_transcript_bytes_per_proposal":1310720,"maximum_witness_transcript_bytes_per_segment":168820736,"no_floating_fallback":true,"operation_policy":"general p/q primitives are FROM_BINARY64,ADD,SUBTRACT,MULTIPLY,DIVIDE,SQUARE,NEGATE,and COMPARE with sign normalization and Euclidean reduction after construction; dyadic primitives are FROM_BINARY64,ADD,SUBTRACT,MULTIPLY,SQUARE,NEGATE,and COMPARE,align ADD/SUBTRACT to the smaller exponent,and canonicalize by shifting every factor of two from m into e; for each dyadic distance guard convert the 3N coordinates once,form each component difference,and accumulate (dx^2+dy^2)+dz^2 left-to-right before one strict comparison with the cached dyadic rho^2; fixed vector-component and canonical-pair order is mandatory in both lanes","representation":"two disjoint custom exact lanes: general values used by the simultaneous first-crossing certificate are canonical built-in-integer pairs (p,q) with q>0,gcd(abs(p),q)=1,and zero (0,1); binary64 duration scheduling and all radius-sum,start,stage,and candidate squared-distance guards use canonical exact dyadics (m,e) denoting m*2^e,with zero (0,0) and every nonzero m odd; finite binary64 enters either lane only through a prospectively authorized float.as_integer_ratio conversion; Fraction,Decimal,third-party exact code,and floating comparison fallback are forbidden","resource_policy":"report general-rational and dyadic deterministic abstract exact-work units separately and their exact sum; the fixed named path weights are exactly ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHTS_V1 and are retained verbatim in the resource specification; a selected named path charges its table weight once at its explicit runtime charge site,including conditional nonzero,even-normalization,nextafter,and conversion paths,while unselected conditional paths charge zero; these are qualification weights for this algorithm,not counts of Python,C,CPU,big-integer,or machine instructions or primitives and not estimates or guarantees of elapsed time,wall time,or memory; each charge site prospectively authorizes its complete table weight against the initialization-plus-segment ledgers during preflight or proposal-plus-segment ledgers during a proposal before the selected path produces its result; explicit integer-width and exponent checks additionally authorize potentially enlarged shift,add,subtract,multiply,and conversion results before those result integers are formed; reduction uses a custom interruptible Euclidean loop only in the general p/q lane and charges exactly one separately reported GCD.EUCLIDEAN_ITERATION unit per selected loop iteration,without claiming a count of underlying modulus or integer primitives; this unit is charged to separate initialization/proposal and segment gcd ledgers without double-charging exact-work units; it also obeys the per-reduction iteration cap; body,pair,and identifier caps are checked before numeric conversion or set allocation,and full initialization and per-proposal operation reservations are required from the remaining segment budget; every normalized rational exponent and integer intermediate is capped; any failed prospective check is fatal and never falls back to opaque math.gcd or floating arithmetic; the dyadic lane performs no GCD","work_weight_table":[["GENERAL_RATIONAL","INTEGER_WIDTH_OBSERVATION",2],["GENERAL_RATIONAL","RATIONAL_EXPONENT_DIAGNOSTIC",1],["GENERAL_RATIONAL","INTEGER_COMPARE",1],["GENERAL_RATIONAL","INTEGER_ABSOLUTE",1],["GENERAL_RATIONAL","INTEGER_NEGATE",1],["GENERAL_RATIONAL","INTEGER_SHIFT_LEFT",1],["GENERAL_RATIONAL","INTEGER_ADD",1],["GENERAL_RATIONAL","INTEGER_SUBTRACT",1],["GENERAL_RATIONAL","INTEGER_MULTIPLY",1],["GENERAL_RATIONAL","INTEGER_FLOOR_DIVIDE",1],["GENERAL_RATIONAL","BINARY64_ENVELOPE_BASE",8],["GENERAL_RATIONAL","BINARY64_ENVELOPE_NONZERO",4],["GENERAL_RATIONAL","RATIONAL_CONSTRUCTION",1],["GENERAL_RATIONAL","FROM_BINARY64_RATIO_EXTRACTION",1],["GENERAL_RATIONAL","SIGNED_STEP_ABSOLUTE",1],["DYADIC","INTEGER_WIDTH_OBSERVATION",2],["DYADIC","BINARY64_ENVELOPE_BASE",8],["DYADIC","BINARY64_ENVELOPE_NONZERO",4],["DYADIC","DYADIC_CONSTRUCTION",1],["DYADIC","INPUT_EXPONENT_CHECK",1],["DYADIC","NONZERO_ODDNESS_BRANCH",1],["DYADIC","EVEN_LOWBIT_PATH",3],["DYADIC","EVEN_CANONICALIZE_PATH",3],["DYADIC","CANONICAL_EXPONENT_DIAGNOSTIC",4],["DYADIC","FROM_BINARY64_RATIO_EXTRACTION",1],["DYADIC","FROM_BINARY64_POWER_OF_TWO_CHECK",2],["DYADIC","ADD_OR_SUBTRACT_ALIGN",3],["DYADIC","ADD_OR_SUBTRACT_SHIFT_PAIR",2],["DYADIC","ADD_OR_SUBTRACT_COMBINE",1],["DYADIC","MULTIPLY_MANTISSA_AND_EXPONENT",2],["DYADIC","NEGATE",1],["DYADIC","COMPARE_ALIGN_AND_SHIFT_PAIR",5],["DYADIC","COMPARE_RESULT",1],["DYADIC","TO_BINARY64_RATIO_BUILD",1],["DYADIC","TO_BINARY64_ROUND",1],["DYADIC","DURATION_ABSOLUTE",1],["DYADIC","SCHEDULER_NEXTAFTER_DOWN",1],["DYADIC","SCHEDULER_SUCCESSOR_PROBE",1],["GCD","EUCLIDEAN_ITERATION",1]],"work_weight_table_id":"ENCOUNTER_EXACT_ABSTRACT_WORK_WEIGHTS_V1"}},"exact_witness_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_JSON_V1","exact_witness_checksum_domain":"jxplanetx.encounter-exact-clearance-witness.content-integrity.v1","exactly_time_reversible":false,"first_crossing_certificate":"for every canonical pair simultaneously require the strict exact-rational inequality ell_ij^2 > (rho_ij+(1/2)*(A_i+A_j)*Delta^2)^2; if an earliest first crossing existed, all pair floors would hold beforehand, the acceleration bounds would bound the relative Taylor remainder by (1/2)*(A_i+A_j)*u^2, and the strict inequality would contradict contact at that crossing; equality is uncertified and is rejected","force_evaluation_accounting":"record per proposal its disposition and actual derivative/force calls; certificate rejection has zero; stage-guard abort has zero through twelve; derivative-domain abort has one through thirteen; completed RK attempt has exactly thirteen finite derivative calls; total force evaluations equal the exact sum of per-proposal calls,not thirteen times all proposals","force_plan_scope":"AUTONOMOUS_MUTUAL_ALL_ACTIVE_POSITIVE_GM_UNSOFTENED_NEWTONIAN_ONLY","globally_symplectic":false,"gm_centroid_error_scale":"C_x=sum_i(GM_i*x_i)/sum_i(GM_i) and C_v analogously in immutable body order; centroid defect is sum_i(GM_i*trial.position_defect_i)/sum_i(GM_i) or the velocity-defect analogue directly from the frozen tableau and scale is centroid_atol+centroid_rtol*max(abs(current_centroid_component),abs(accepted_centroid_component)); position and velocity controls are separate and all reductions use fixed left-to-right NumPy float64 order","gm_centroid_position_atol":{"float_hex":"0x1.0c6f7a0b5ed8dp-20"},"gm_centroid_position_rtol":{"float_hex":"0x1.19799812dea11p-40"},"gm_centroid_velocity_atol":{"float_hex":"0x1.0c6f7a0b5ed8dp-20"},"gm_centroid_velocity_rtol":{"float_hex":"0x1.19799812dea11p-40"},"initial_domain_policy":"before any force call require the accepted initial numerical node finite and,for every canonical pair,the exact-dyadic squared separation strictly greater than rho_ij^2; failure is fatal and returns no result","initial_epoch":{"float_hex":"0x0.0p+0"},"initial_step_magnitude":{"float_hex":"0x1.0000000000000p-3"},"initialization_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_JSON_V1","initialization_checksum_domain":"jxplanetx.encounter-initialization.content-integrity.v1","initialization_resource_policy":"before proposal one,charge every exact GM,radius,floor,and initial-position conversion,every rho_ij>=radius_i+radius_j proof,every initial squared-separation>rho_ij^2 proof,and construction of all cached A_i=sum_{k!=i}(GM_k/rho_ik^2) values to a separately capped initialization dyadic-work,general-rational-work,gcd,transcript ledger and also to the enclosing lane\'s whole-segment ledgers; retain a domain-separated initialization transcript digest plus diagnostics bounded by maximum_witness_diagnostic_bytes_per_proposal; successful work is never hidden in a segment-only pool or charged to proposal one,while any initialization failure is fatal and releases no partial result; mandatory replay recomputes initialization under an identical separate ledger","input_binding_policy":"snapshot body_ids must exactly equal body_order,snapshot.epoch must be an exact built-in float bit-identical to initial_epoch,and metadata must name the required frame,origin,allowed axes,and allowed time scale; positions,velocities,GM,masses,and radii must be exact CPU numpy.ndarray objects with float64 dtype,required shapes,C-contiguous finite storage,and no subclass or device transfer; massive must be an exact CPU NumPy boolean array containing only true; every GM is strictly positive,every radius is nonnegative,and physical masses are retained but unused by the GM-scaled dynamics; the force plan is exactly one NewtonianPointMass with source_ids=target_ids=body_order on numpy.cpu and contains no other force; all parameter-metadata validity intervals must cover the closed public label interval between initial_epoch and endpoint_epoch","linear_minimum_policy":"for one proposed signed step h_s set Delta=abs(h_s), r_ij=x_j-x_i, w_ij=sign(h_s)*(v_j-v_i), b=dot(r_ij,w_ij), c=dot(w_ij,w_ij),and q=dot(r_ij,r_ij), all exactly; the exact minimum ell^2 of norm(r_ij+u*w_ij)^2 on 0<=u<=Delta is q when c=0 or b>=0, norm(r_ij+Delta*w_ij)^2 when b<0 and -b>=c*Delta, and q-b^2/c otherwise; branch comparisons and ell^2 are exact rational","maximum_accepted_substeps":128,"maximum_consecutive_rejections":32,"maximum_force_evaluations":1664,"maximum_rejected_substeps":64,"maximum_scale_factor":{"float_hex":"0x1.4000000000000p+2"},"maximum_step_magnitude":{"float_hex":"0x1.0000000000000p-3"},"maximum_substep_proposals":128,"method_class":"EXPLICIT_EMBEDDED_RUNGE_KUTTA_LOCAL_ENCOUNTER_IVP","method_id":"integrator.adaptive.rkf78.encounter_segment_newtonian_v1","minimum_scale_factor":{"float_hex":"0x1.999999999999ap-3"},"minimum_step_failure_policy":"a proposal rejected at minimum_step_magnitude,an exact remaining chunk below that magnitude except a deterministic endpoint-remainder chunk,or a rejected terminal endpoint-remainder chunk strictly below minimum-step magnitude,or any exhausted proposal,acceptance,rejection,force,or exact-resource cap is fatal; an ordinary clipped remainder strictly greater than minimum may reject and retry or split,while equality is fatal on rejection","minimum_step_magnitude":{"float_hex":"0x1.0624dd2f1a9fcp-10"},"pair_certification_floors":[{"float_hex":"0x1.47ae147ae147bp-7"}],"pair_error_scale":"for pair i<j and component k, defect is (trial.position_defect_j-trial.position_defect_i)_k or (trial.velocity_defect_j-trial.velocity_defect_i)_k directly from the frozen Fehlberg tableau result,never recomputed from Kahan-updated accepted minus embedded states; scale is pair_atol_ij+pair_rtol*max(abs(current_j-current_i)_k,abs(accepted_j-accepted_i)_k); apply separately to position and velocity","pair_position_atols":[{"float_hex":"0x1.0c6f7a0b5ed8dp-20"}],"pair_position_rtol":{"float_hex":"0x1.19799812dea11p-40"},"pair_table_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1","pair_table_checksum_domain":"jxplanetx.encounter-pair-table.content-integrity.v1","pair_table_policy":"for N bodies the canonical table has exactly N*(N-1)/2 entries ordered (0,1),(0,2),...,(0,N-1),(1,2),...,(N-2,N-1); each supplied binary64 certification floor is positive, retained verbatim, and must be exact-dyadically greater than or equal to the exact sum of the two retained nonnegative binary64 radii; no scalar or hidden default floor","pair_velocity_atols":[{"float_hex":"0x1.0c6f7a0b5ed8dp-20"}],"pair_velocity_rtol":{"float_hex":"0x1.19799812dea11p-40"},"principal_order":8,"proposal_accounting":"proposals=certificate_rejections+stage_guard_aborts+derivative_domain_aborts+completed_rk_attempts; completed_rk_attempts=candidate_domain_rejections+error_rejections+accepted_substeps; rejected_substeps=certificate_rejections+stage_guard_aborts+derivative_domain_aborts+candidate_domain_rejections+error_rejections; total and consecutive rejection caps apply to this exact five-category sum; all counts are exact nonnegative built-in integers","public_execution_accounting_scope":"PRIMARY_PLUS_MANDATORY_SEMANTIC_REPLAY_WITH_SEPARATE_AND_TOTAL_COUNTERS","qualification_authorized":false,"registry_authorized":false,"required_frame":"BARYCENTRIC_INERTIAL","required_origin":"BARYCENTER","resource_lane_policy":"primary execution and mandatory semantic replay each receive separate identical proposal,acceptance,rejection,force,body,pair,integer,exponent,gcd,operation,transcript,and witness caps,including a separate initialization ledger in each lane; report initialization,proposal,lane-total,primary,replay,and public-total usage separately; the public-total force,exact-operation,GCD,and transcript ceilings are exactly twice the corresponding one-lane ceilings; neither lane may borrow unused budget from the other","resource_nonclaim_policy":"fixed logical caps make arithmetic and custody finite but are not a wall-clock or resident-memory denial-of-service guarantee; an external service must additionally enforce process isolation,time,and memory limits","result_content_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1","result_content_checksum_domain":"jxplanetx.encounter-segment-result.content-integrity.v1","result_custody_policy":"result arrays and ledgers are owned immutable copies bound to input provenance,body order,pair tables,tolerances,resources,duration,schedule,primary accounting,and replay accounting by domain-separated checksums","safety_factor":{"float_hex":"0x1.ccccccccccccdp-1"},"schedule_checksum_algorithm":"SHA256_DOMAIN_SEPARATED_FLOAT_HEX_JSON_V1","schedule_checksum_domain":"jxplanetx.encounter-segment-schedule.content-integrity.v1","source_author":"Erwin Fehlberg","source_document_id":"19680027281","source_publication_date":"1968-10-01","source_report":"NASA-TR-R-287","source_title":"Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta Formulas with Stepsize Control","source_url":"https://ntrs.nasa.gov/citations/19680027281","stage_abort_policy":"a failed pre-derivative stage guard aborts that RK attempt immediately and retains the actual number from zero through twelve of derivative/force calls already completed; a nonfinite derivative is a distinct derivative-domain abort retaining the actual number from one through thirteen of calls made; only thirteen finite derivatives constitute a completed RK attempt,and candidate-domain and error rejections exist only after such a completed attempt","stage_count":13,"stage_epoch_policy":"CONSTANT_INITIAL_PUBLIC_LABEL_FOR_ALL_AUTONOMOUS_NEWTONIAN_STAGE_EVALUATIONS; every derivative evaluation presents the bit-identical initial_epoch public label to force metadata and ignores the RK stage-time label; exact local offsets,accepted signed substeps,proposal signed steps,and the fixed RKF78 c-table identify mathematical stage times separately; the full closed public label interval between initial_epoch and endpoint_epoch remains subject to parameter-metadata preflight validation; no public label alters or supplies a retained RK substep or force","stage_guard_policy":"before each of the 13 derivative calls require every trial-stage position and velocity finite and exact-dyadic squared pair separation strictly greater than rho_ij^2; after a completed 13-stage attempt apply the same finite strict guard to the eighth-order candidate before error acceptance; a trial-local nonfinite stage,candidate,defect,derivative,or a stage/candidate at or below a floor is not an event and is a recoverable fixed-factor domain rejection; accepted-state nonfiniteness,initial floor failure,or exact-resource failure is fatal","step_scheduler_policy":"proposals have sign(h); set exact positive target to min(exact remaining duration,exact proposed positive binary64 magnitude,exact maximum-step magnitude); convert target once by correctly-rounded nearest-ties-to-even binary64,then if that value is greater than target by exact-dyadic comparison apply math.nextafter(value,0.0) once; require the result finite and positive,which is the unique largest positive binary64 no greater than target; exact-subtract each accepted magnitude from remaining and repeat until exact zero; every chunk remains subject to the certificate,stage guards,and error test; a clipped chunk strictly greater than minimum may reject and retry/split normally,while a terminal chunk <minimum is acceptance-only and any rejection is fatal; fail closed if conversion,progress,exact subtraction,or endpoint completion violates a step or resource cap","tableau_id":"NASA_TR_R_287_FEHLBERG_7_8_13_STAGE","time_policy":"retain initial_epoch,an externally supplied integrity-bound but unauthenticated provenance-only endpoint_epoch label,and one nonzero signed binary64 mathematical duration h; only finiteness and strict monotonicity from initial_epoch in sign(h) are internally checkable,so endpoint_epoch need not equal float(initial_epoch+h) and is not evidence of outer-lattice construction; advance state on an authoritative local offset lattice from exact zero to exact h and never recover any RK step or state duration from endpoint_epoch-initial_epoch; accepted signed binary64 substeps are exact-rationally accounted and their mathematical sum must equal h exactly","transaction_policy":"copy validated inputs to owned work buffers before evaluation; never mutate caller buffers; keep accepted state,carries,local offset,and exact duration ledger unchanged until certificate,all stages,candidate guard,and error acceptance succeed; discard every rejected trial completely; on any fatal failure return no partial result","validation_replay_count":1,"validation_replay_policy":"one full deterministic replay from owned input recomputes adaptive decisions,exact certificates,stage guards,states,carries,local offsets,proposal dispositions,actual force counts,and witness hashes; require bitwise-identical arrays and identical scalar ledgers before release; resource caps and usage obey the separate-lane resource policy","witness_custody_policy":"retain per proposal only canonical pair/disposition identifiers,actual force-call count,bounded binary64 diagnostics,bounded integer bit-length diagnostics,and a domain-separated SHA-256 of the canonical exact comparison transcript; never retain unbounded rational numerators or denominators; replay recomputes every exact inequality","witness_serialization_policy":"for initialization and each proposal serialize UTF-8 JSON with sort_keys=True,ensure_ascii=True,separators=(\',\',\':\'),allow_nan=False; payload contains method/version,phase,proposal index or null,signed-step float.hex or null,disposition,actual force calls,canonical pair indices,linear-minimum branch,and every exact comparison operand as either canonical base-10 numerator and positive denominator strings or canonical base-10 dyadic mantissa and built-in integer exponent; charge and prospectively cap the decimal and JSON byte envelope before conversion,serialization,retention,or hashing,and then charge every actual UTF-8 transcript byte before streaming domain+\'\\\\0\'+payload into SHA-256; retain only the digest and bounded diagnostics,not the serialized exact operands"}},"streamed_witness_ledger_bytes":23393,"transcript_byte_count":205772}},"probe_committed":false,"signed_step":{"float_hex":"0x1.0000000000000p-3"},"start_epoch":{"float_hex":"0x0.0p+0"}}}],"schema":"jxplanetx.hybrid-wh-rkf78-step-ledger.payload.v1"}', 28255, 'e444fec787a1928b6f1ef412ae263540b7963b947281239b890bef8a2a1a40a9')


class HybridRuntimeTests(unittest.TestCase):
    def test_all_private_component_and_step_ledger_complete_literal_preimages(self):
        state, plan, spec = hybrid_fixture(
            steps=1, wh_floor=0.9, periapse_floor=0.5
        )
        result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
            state, plan, spec
        )
        child = result.outer_step_records[0].private_encounter
        self.assertIsNotNone(child)
        assert child is not None
        segment_spec = _bind_encounter_segment(
            profile=spec.encounter_control,
            body_order=spec.body_order,
            initial_epoch=result.outer_epochs[0],
            endpoint_epoch=result.outer_epochs[1],
            duration=spec.wisdom_holman_spec.fixed_step,
        )
        run = encounter_runtime._execute(state, plan, segment_spec)
        domains = {
            field_name: domain
            for field_name, domain, _description in (
                HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER
            )
        }
        schemas = {
            field_name: schema
            for field_name, schema, _keys, _description in (
                HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER
            )
        }

        def ordinary_preimage(domain, payload):
            return domain.encode("utf-8") + b"\x00" + encounter_runtime._canonical_json(
                payload
            )

        def sequence_preimage(domain, schema, values):
            framed = (
                domain.encode("utf-8")
                + b"\x00"
                + schema.encode("utf-8")
                + b"\x00"
                + len(values).to_bytes(8, "big", signed=False)
            )
            for value in values:
                encoded = encounter_runtime._canonical_json(value)
                framed += len(encoded).to_bytes(8, "big", signed=False) + encoded
            return framed

        summary = {
            name: getattr(child, name)
            for name, _source in HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER
        }
        component_digests = {
            name: getattr(child, name)
            for name in HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER[:-1]
        }
        actual = {
            "final_state_content_sha256": ordinary_preimage(
                domains["final_state_content_sha256"],
                {
                    "schema": schemas["final_state_content_sha256"],
                    "positions": run.final_positions,
                    "velocities": run.final_velocities,
                },
            ),
            "initialization_content_sha256": ordinary_preimage(
                domains["initialization_content_sha256"],
                {
                    "schema": schemas["initialization_content_sha256"],
                    "record": run.initialization_record,
                },
            ),
            "proposal_ledger_content_sha256": sequence_preimage(
                domains["proposal_ledger_content_sha256"],
                schemas["proposal_ledger_content_sha256"],
                run.proposal_ledger,
            ),
            "force_ledger_content_sha256": sequence_preimage(
                domains["force_ledger_content_sha256"],
                schemas["force_ledger_content_sha256"],
                run.force_ledger,
            ),
            "primary_counts_content_sha256": ordinary_preimage(
                domains["primary_counts_content_sha256"],
                {
                    "schema": schemas["primary_counts_content_sha256"],
                    "counts": run.counts,
                },
            ),
            "accepted_steps_content_sha256": sequence_preimage(
                domains["accepted_steps_content_sha256"],
                schemas["accepted_steps_content_sha256"],
                run.accepted_signed_substeps,
            ),
            "private_execution_content_sha256": ordinary_preimage(
                domains["private_execution_content_sha256"],
                {
                    "schema": schemas["private_execution_content_sha256"],
                    "method_id": segment_spec.method_id,
                    "outer_step_index": 1,
                    "segment_spec": segment_spec,
                    "summary": summary,
                    "component_digests": component_digests,
                },
            ),
        }
        self.assertEqual(
            tuple(_PRIVATE_COMPONENT_LITERAL_KATS),
            HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER,
        )
        for name in HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER:
            literal, literal_length, literal_sha256 = (
                _PRIVATE_COMPONENT_LITERAL_KATS[name]
            )
            self.assertIs(type(literal), bytes)
            self.assertEqual(len(literal), literal_length)
            self.assertEqual(hashlib.sha256(literal).hexdigest(), literal_sha256)
            self.assertEqual(actual[name], literal)
            self.assertEqual(getattr(child, name), literal_sha256)

        step_literal, step_length, step_sha256 = _HYBRID_STEP_LEDGER_LITERAL_KAT
        actual_step_preimage = (
            hybrid_runtime.HYBRID_STEP_LEDGER_CHECKSUM_DOMAIN.encode("utf-8")
            + b"\x00"
            + encounter_runtime._canonical_json(
                {
                    "schema": hybrid_runtime._HYBRID_STEP_LEDGER_SCHEMA,
                    "records": result.outer_step_records,
                }
            )
        )
        self.assertIs(type(step_literal), bytes)
        self.assertEqual(len(step_literal), step_length)
        self.assertEqual(hashlib.sha256(step_literal).hexdigest(), step_sha256)
        self.assertEqual(actual_step_preimage, step_literal)
        self.assertEqual(result.step_ledger_content_sha256, step_sha256)

    def test_literal_preimage_required_mutation_matrix_is_complete(self):
        def changed(literal, needle, replacement):
            self.assertIn(needle, literal)
            mutated = literal.replace(needle, replacement, 1)
            self.assertNotEqual(mutated, literal)
            self.assertNotEqual(
                hashlib.sha256(mutated).hexdigest(),
                hashlib.sha256(literal).hexdigest(),
            )

        domain_rows = {
            name: domain
            for name, domain, _description in (
                HYBRID_PRIVATE_ENCOUNTER_DIGEST_DOMAIN_ROSTER
            )
        }
        schema_rows = {
            name: (schema, keys)
            for name, schema, keys, _description in (
                HYBRID_PRIVATE_ENCOUNTER_PREIMAGE_SCHEMA_ROSTER
            )
        }
        for name in HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER:
            literal = _PRIVATE_COMPONENT_LITERAL_KATS[name][0]
            domain = domain_rows[name].encode("utf-8")
            schema, keys = schema_rows[name]
            changed(literal, domain, domain[:-1] + b"X")
            changed(
                literal,
                schema.encode("utf-8"),
                schema.encode("utf-8")[:-1] + b"X",
            )
            if keys != ("sequence",):
                for key in keys:
                    token = b'"' + key.encode("utf-8") + b'":'
                    changed(
                        literal,
                        token,
                        b'"mutated_' + key.encode("utf-8") + b'":',
                    )

        initialization = _PRIVATE_COMPONENT_LITERAL_KATS[
            "initialization_content_sha256"
        ][0]
        changed(
            initialization,
            b"jxplanetx.engine.encounter.EncounterInitializationRecord",
            b"jxplanetx.engine.encounter.MutatedInitializationRecord",
        )
        changed(
            initialization,
            b'"maximum_integer_bits":',
            b'"mutated_maximum_integer_bits":',
        )

        final_state = _PRIVATE_COMPONENT_LITERAL_KATS[
            "final_state_content_sha256"
        ][0]
        changed(final_state, b'"dtype":"float64"', b'"dtype":"float32"')
        changed(final_state, b'"shape":[2,3]', b'"shape":[3,2]')
        values_start = final_state.index(b'"values":[') + len(b'"values":[')
        first_end = final_state.index(b",", values_start)
        second_end = final_state.index(b",", first_end + 1)
        first_value = final_state[values_start:first_end]
        second_value = final_state[first_end + 1 : second_end]
        reordered = (
            final_state[:values_start]
            + second_value
            + b","
            + first_value
            + final_state[second_end:]
        )
        self.assertNotEqual(
            hashlib.sha256(reordered).hexdigest(),
            hashlib.sha256(final_state).hexdigest(),
        )

        signed_zero_literal = (
            b"jxplanetx.hybrid-private-encounter-accepted-steps.content-integrity.v1\x00"
            b"jxplanetx.hybrid-private-encounter-accepted-steps.sequence.v1\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x02"
            b"\x00\x00\x00\x00\x00\x00\x00\x24"
            b'{"float_hex":"0x1.0000000000000p-1"}'
            b"\x00\x00\x00\x00\x00\x00\x00\x19"
            b'{"float_hex":"-0x0.0p+0"}'
        )
        self.assertEqual(len(signed_zero_literal), 218)
        self.assertEqual(
            hashlib.sha256(signed_zero_literal).hexdigest(),
            "79908ca6530f7492121a6c140c5278c36c3123ab552165fa022b2cb96fca7ae5",
        )
        changed(
            signed_zero_literal,
            b'"-0x0.0p+0"',
            b'"0x0.0p+0"',
        )

        proposal = _PRIVATE_COMPONENT_LITERAL_KATS[
            "proposal_ledger_content_sha256"
        ][0]
        proposal_prefix = (
            domain_rows["proposal_ledger_content_sha256"].encode("utf-8")
            + b"\x00"
            + schema_rows["proposal_ledger_content_sha256"][0].encode("utf-8")
            + b"\x00"
        )
        count_offset = len(proposal_prefix)
        count = int.from_bytes(proposal[count_offset : count_offset + 8], "big")
        self.assertGreaterEqual(count, 2)
        changed(
            proposal,
            proposal[count_offset : count_offset + 8],
            (count - 1).to_bytes(8, "big"),
        )
        first_length_offset = count_offset + 8
        first_length = int.from_bytes(
            proposal[first_length_offset : first_length_offset + 8], "big"
        )
        changed(
            proposal,
            proposal[first_length_offset : first_length_offset + 8],
            (first_length + 1).to_bytes(8, "big"),
        )
        first_start = first_length_offset
        first_stop = first_start + 8 + first_length
        second_length = int.from_bytes(proposal[first_stop : first_stop + 8], "big")
        second_stop = first_stop + 8 + second_length
        reordered_items = (
            proposal[:first_start]
            + proposal[first_stop:second_stop]
            + proposal[first_start:first_stop]
            + proposal[second_stop:]
        )
        self.assertNotEqual(
            hashlib.sha256(reordered_items).hexdigest(),
            hashlib.sha256(proposal).hexdigest(),
        )

        manifest = _PRIVATE_COMPONENT_LITERAL_KATS[
            "private_execution_content_sha256"
        ][0]
        for summary_name, _source in HYBRID_PRIVATE_ENCOUNTER_SUMMARY_SOURCE_ROSTER:
            token = b'"' + summary_name.encode("utf-8") + b'":'
            changed(
                manifest,
                token,
                b'"mutated_' + summary_name.encode("utf-8") + b'":',
            )
            value_start = manifest.index(token) + len(token)
            value_stop = value_start
            while manifest[value_stop : value_stop + 1] in b"0123456789":
                value_stop += 1
            source_value = int(manifest[value_start:value_stop])
            value_mutation = (
                manifest[:value_start]
                + str(source_value + 1).encode("ascii")
                + manifest[value_stop:]
            )
            self.assertNotEqual(
                hashlib.sha256(value_mutation).hexdigest(),
                hashlib.sha256(manifest).hexdigest(),
            )
        for component_name in HYBRID_PRIVATE_ENCOUNTER_DIGEST_FIELD_ROSTER[:-1]:
            token = b'"' + component_name.encode("utf-8") + b'":'
            changed(
                manifest,
                token,
                b'"mutated_' + component_name.encode("utf-8") + b'":',
            )
        manifest_payload = manifest.split(b"\x00", 1)[1]
        self.assertNotIn(
            b'"private_execution_content_sha256":', manifest_payload
        )
        self_including = manifest.replace(
            b'"component_digests":{',
            b'"component_digests":{"private_execution_content_sha256":"'
            + b"0" * 64
            + b'",',
            1,
        )
        self.assertNotEqual(
            hashlib.sha256(self_including).hexdigest(),
            hashlib.sha256(manifest).hexdigest(),
        )

        step_literal = _HYBRID_STEP_LEDGER_LITERAL_KAT[0]
        for identity in (
            b"jxplanetx.engine.hybrid.HybridOuterStepRecord",
            b"jxplanetx.engine.hybrid_contracts.HybridFarProbeDecision",
            b"jxplanetx.engine.hybrid_contracts.HybridFarProbeWork",
            b"jxplanetx.engine.hybrid_contracts.HybridKeplerProbeRecord",
            b"jxplanetx.engine.hybrid_contracts.HybridPrivateEncounterRecord",
        ):
            changed(step_literal, identity, identity + b"Mutated")
        field_groups = (
            tuple(field.name for field in dataclasses.fields(HybridOuterStepRecord)),
            tuple(field.name for field in dataclasses.fields(HybridFarProbeDecision)),
            HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER
            + ("kepler_probe_records",),
            tuple(field.name for field in dataclasses.fields(HybridKeplerProbeRecord)),
            tuple(field.name for field in dataclasses.fields(HybridPrivateEncounterRecord)),
        )
        for group in field_groups:
            for field_name in group:
                token = b'"' + field_name.encode("utf-8") + b'":'
                changed(
                    step_literal,
                    token,
                    b'"mutated_' + field_name.encode("utf-8") + b'":',
                )

    def test_private_streaming_sequence_has_independent_literal_known_answer(self):
        domain = (
            "jxplanetx.hybrid-private-encounter-accepted-steps."
            "content-integrity.v1"
        )
        schema = (
            "jxplanetx.hybrid-private-encounter-accepted-steps.sequence.v1"
        )
        literal = (
            b"jxplanetx.hybrid-private-encounter-accepted-steps.content-integrity.v1\x00"
            b"jxplanetx.hybrid-private-encounter-accepted-steps.sequence.v1\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x02"
            b"\x00\x00\x00\x00\x00\x00\x00\x24"
            b'{"float_hex":"0x1.0000000000000p-1"}'
            b"\x00\x00\x00\x00\x00\x00\x00\x19"
            b'{"float_hex":"-0x0.0p+0"}'
        )
        self.assertEqual(len(literal), 218)
        self.assertEqual(
            hashlib.sha256(literal).hexdigest(),
            "79908ca6530f7492121a6c140c5278c36c3123ab552165fa022b2cb96fca7ae5",
        )
        self.assertEqual(
            hybrid_runtime._streaming_sequence_sha256(
                domain=domain,
                schema=schema,
                values=(0.5, -0.0),
                max_count=2,
                max_item_bytes=40,
                max_cumulative_bytes=hybrid_runtime._streaming_cumulative_cap(
                    domain=domain,
                    schema=schema,
                    max_count=2,
                    max_item_payload_bytes=80,
                ),
            ),
            hashlib.sha256(literal).hexdigest(),
        )

    def test_streaming_and_witness_byte_caps_are_checked_prospectively(self):
        domain = "jxplanetx.test.hybrid-streaming.v1"
        schema = "jxplanetx.test.hybrid-streaming.sequence.v1"
        values = (0.5, -0.0)
        exact = (
            domain.encode("utf-8")
            + b"\x00"
            + schema.encode("utf-8")
            + b"\x00"
            + len(values).to_bytes(8, "big")
        )
        for value in values:
            encoded = encounter_runtime._canonical_json(value)
            exact += len(encoded).to_bytes(8, "big") + encoded
        reserved_cap = hybrid_runtime._streaming_cumulative_cap(
            domain=domain,
            schema=schema,
            max_count=2,
            max_item_payload_bytes=80,
        )
        self.assertEqual(
            hybrid_runtime._streaming_sequence_sha256(
                domain=domain,
                schema=schema,
                values=values,
                max_count=2,
                max_item_bytes=40,
                max_cumulative_bytes=reserved_cap,
            ),
            hashlib.sha256(exact).hexdigest(),
        )
        with mock.patch.object(
            encounter_runtime,
            "_canonical_json",
            side_effect=AssertionError("count rejection traversed an item"),
        ):
            with self.assertRaises(HybridContractError):
                hybrid_runtime._streaming_sequence_sha256(
                    domain=domain,
                    schema=schema,
                    values=values,
                    max_count=1,
                    max_item_bytes=40,
                    max_cumulative_bytes=reserved_cap,
                )
        with self.assertRaises(HybridContractError):
            hybrid_runtime._streaming_sequence_sha256(
                domain=domain,
                schema=schema,
                values=values,
                max_count=2,
                max_item_bytes=len(encounter_runtime._canonical_json(0.5)) - 1,
                max_cumulative_bytes=reserved_cap,
            )
        with self.assertRaises(HybridContractError):
            hybrid_runtime._streaming_sequence_sha256(
                domain=domain,
                schema=schema,
                values=values,
                max_count=2,
                max_item_bytes=40,
                max_cumulative_bytes=(
                    len(domain.encode("utf-8"))
                    + 1
                    + len(schema.encode("utf-8"))
                    + 1
                    + 8
                    + 8
                    + len(encounter_runtime._canonical_json(values[0]))
                    + 8
                    + 40
                    - 1
                ),
            )
        preamble_and_one_frame = (
            len(domain.encode("utf-8"))
            + 1
            + len(schema.encode("utf-8"))
            + 1
            + 8
            + 8
        )
        for item_cap in (0, 40):
            with mock.patch.object(
                encounter_runtime,
                "_canonical_json",
                side_effect=AssertionError("encoder called without reserved capacity"),
            ):
                with self.assertRaises(HybridContractError):
                    hybrid_runtime._streaming_sequence_sha256(
                        domain=domain,
                        schema=schema,
                        values=(0.5,),
                        max_count=1,
                        max_item_bytes=item_cap,
                        max_cumulative_bytes=preamble_and_one_frame,
                    )
        encoded = encounter_runtime._canonical_json(0.5)
        self.assertEqual(
            hybrid_runtime._bounded_canonical_json(
                0.5,
                max_item_bytes=len(encoded),
                remaining_cumulative_bytes=len(encoded),
                label="KAT witness",
            ),
            encoded,
        )
        for item_cap, cumulative_cap in (
            (len(encoded) - 1, len(encoded)),
            (len(encoded), len(encoded) - 1),
        ):
            with self.assertRaises(HybridContractError):
                hybrid_runtime._bounded_canonical_json(
                    0.5,
                    max_item_bytes=item_cap,
                    remaining_cumulative_bytes=cumulative_cap,
                    label="KAT witness",
                )

    def test_private_child_seven_digest_fixture_known_answers(self):
        state, plan, spec = hybrid_fixture(
            steps=1, wh_floor=0.9, periapse_floor=0.5
        )
        result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
            state, plan, spec
        )
        child = result.outer_step_records[0].private_encounter
        self.assertIsNotNone(child)
        assert child is not None
        expected = {
            "final_state_content_sha256": (
                "816c7feb2cad928c1609a2c7a6cdd9cde1eb377347b01b80391e0be8da72c4e4"
            ),
            "initialization_content_sha256": (
                "fc818f80229f200f1d6db6d03f3020b3cf6e3336fff9536c0d90a82e65a5b66e"
            ),
            "proposal_ledger_content_sha256": (
                "c30405f0f3bce64c3b358a0d37706dde57b96ef8c883d6aa4c96081fbc8b414f"
            ),
            "force_ledger_content_sha256": (
                "db73abb2022b465e7b63f828e6f183cc5a9dbfdb36fc2004f11f742639e7cab5"
            ),
            "primary_counts_content_sha256": (
                "bf733a488083fd9f966b1e1a977f2ac437439f2ab92d904a9686f6a8e193f8d3"
            ),
            "accepted_steps_content_sha256": (
                "8c9bee865756515a47dec1ad776f3c1fcd25bff2a7c61f206ec65a0fa4211631"
            ),
            "private_execution_content_sha256": (
                "8a0235b549878a5cb75b22872d2c434c5f14b6f5f051abbcbc94ecc799f44789"
            ),
        }
        self.assertEqual(
            {name: getattr(child, name) for name in expected}, expected
        )

    def test_all_far_projects_bit_exact_public_wh_forward_and_backward(self):
        for signed_step in (0.125, -0.125):
            for checkpoint_roster in ((0, 3), (0, 1, 2, 3)):
                with self.subTest(
                    signed_step=signed_step, checkpoints=checkpoint_roster
                ):
                    state, plan, spec = hybrid_fixture(
                        steps=3,
                        fixed_step=signed_step,
                        checkpoints=checkpoint_roster,
                    )
                    hybrid = integrate_hybrid_wisdom_holman_rkf78_trajectory(
                        state, plan, spec
                    )
                    wh = integrate_wisdom_holman_trajectory(
                        state, plan, spec.wisdom_holman_spec
                    )
                    self.assertTrue(hybrid.all_far)
                    self.assertEqual(
                        hybrid.wisdom_holman_schedule_content_sha256,
                        wh.schedule_content_sha256,
                    )
                    self.assertEqual(
                        hybrid.wisdom_holman_result_content_sha256,
                        wh.result_content_sha256,
                    )
                    for observed, expected in zip(hybrid.checkpoints, wh.checkpoints):
                        self.assertEqual(
                            observed.positions.tobytes(order="C"),
                            expected.positions.tobytes(order="C"),
                        )
                        self.assertEqual(
                            observed.velocities.tobytes(order="C"),
                            expected.velocities.tobytes(order="C"),
                        )
                    work = hybrid.primary_counts.accepted_wh_work
                    expected_counts = {
                    "force_evaluations_completed": wh.primary_map_force_evaluations,
                    "interaction_force_assemblies": (
                        wh.primary_map_interaction_force_assemblies
                    ),
                    "kepler_subflow_solves_completed": (
                        wh.primary_map_kepler_subflow_solves
                    ),
                    "universal_solver_iterations": (
                        wh.primary_map_universal_solver_iterations
                    ),
                    "universal_solver_bracket_expansions": (
                        wh.primary_map_universal_solver_bracket_expansions
                    ),
                    "universal_g_bundle_calls_completed": (
                        wh.primary_map_universal_g_function_evaluations
                    ),
                    "universal_series_terms_evaluated": (
                        wh.primary_map_universal_series_terms_evaluated
                    ),
                    "cartesian_to_jacobi_transforms_completed": (
                        wh.primary_map_coordinate_forward_transforms
                    ),
                    "jacobi_to_cartesian_transforms_completed": (
                        wh.primary_map_coordinate_inverse_transforms
                    ),
                    "node_guard_evaluations": wh.primary_map_node_guard_evaluations,
                    "path_guard_evaluations": wh.primary_map_path_guard_evaluations,
                    }
                    for name, expected in expected_counts.items():
                        self.assertEqual(getattr(work, name), expected)
                    exact_flow = {
                        "force_calls_entered": 4,
                        "force_evaluations_completed": 4,
                        "interaction_force_assemblies": 4,
                        "kepler_subflow_calls_entered": 3,
                        "kepler_subflow_solves_completed": 3,
                        "cartesian_to_jacobi_calls_entered": 1,
                        "cartesian_to_jacobi_transforms_completed": 1,
                        "jacobi_to_cartesian_calls_entered": 6,
                        "jacobi_to_cartesian_transforms_completed": 6,
                        "node_guard_evaluations": 7,
                        "path_guard_evaluations": 3,
                        "first_half_kicks_completed": 3,
                        "center_of_mass_drifts_completed": 3,
                        "second_half_kicks_completed": 3,
                    }
                    for name, expected in exact_flow.items():
                        self.assertEqual(getattr(work, name), expected)
                    diagnostics = hybrid.wisdom_holman_diagnostics
                    self.assertIsNotNone(diagnostics)
                    assert diagnostics is not None
                    for descriptor in dataclasses.fields(diagnostics):
                        self.assertEqual(
                            encounter_runtime._canonical_json(
                                getattr(diagnostics, descriptor.name)
                            ),
                            encounter_runtime._canonical_json(
                                getattr(wh, descriptor.name)
                            ),
                        )

    def test_real_path_near_discards_probe_and_redoes_original_interval(self):
        state, plan, spec = hybrid_fixture(
            steps=1,
            wh_floor=0.9,
            periapse_floor=0.5,
            atol=1.0e-6,
        )
        input_position_bytes = state.positions.tobytes(order="C")
        input_velocity_bytes = state.velocities.tobytes(order="C")
        wh_force_epochs: list[float] = []
        original_wh_evaluator = wh_runtime.evaluate_force_plan

        def counted_wh_evaluator(snapshot, force_plan):
            wh_force_epochs.append(float(snapshot.epoch))
            return original_wh_evaluator(snapshot, force_plan)

        with mock.patch.object(
            wh_runtime,
            "evaluate_force_plan",
            side_effect=counted_wh_evaluator,
        ):
            result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
                state, plan, spec
            )
        self.assertFalse(result.all_far)
        record = result.outer_step_records[0]
        self.assertEqual(record.mode, "CARTESIAN_RKF78_NEAR_FULL_INTERVAL")
        self.assertEqual(record.decision.outcome, "NEAR_SWITCH")
        self.assertEqual(
            record.decision.phase, "POST_DRIFT_PRE_FORCE_PATH_SCREEN"
        )
        self.assertEqual(record.decision.pair_indices, ((0, 1),))
        self.assertFalse(record.probe_committed)
        self.assertEqual(record.decision.work.force_calls_entered, 1)
        self.assertEqual(record.decision.work.force_evaluations_completed, 1)
        self.assertEqual(record.decision.work.interaction_force_assemblies, 1)
        self.assertEqual(record.decision.work.path_guard_evaluations, 1)
        self.assertEqual(record.decision.work.second_half_kicks_completed, 0)
        self.assertEqual(wh_force_epochs, [0.0, 0.0])
        self.assertNotIn(0.125, wh_force_epochs)
        self.assertIsNotNone(record.private_encounter)

        child_spec = _bind_encounter_segment(
            profile=result.integration_spec.encounter_control,
            body_order=result.integration_spec.body_order,
            initial_epoch=result.outer_epochs[0],
            endpoint_epoch=result.outer_epochs[1],
            duration=result.integration_spec.wisdom_holman_spec.fixed_step,
        )
        child_run = encounter_runtime._execute(
            result.initial_snapshot, result.force_plan, child_spec
        )
        self.assertEqual(
            result.final_positions.tobytes(order="C"),
            child_run.final_positions.tobytes(order="C"),
        )
        self.assertEqual(
            result.final_velocities.tobytes(order="C"),
            child_run.final_velocities.tobytes(order="C"),
        )
        expected_record = hybrid_runtime._private_encounter_record(
            outer_step_index=1, segment_spec=child_spec, run=child_run
        )
        self.assertEqual(
            encounter_runtime._canonical_json(record.private_encounter),
            encounter_runtime._canonical_json(expected_record),
        )
        self.assertEqual(state.positions.tobytes(order="C"), input_position_bytes)
        self.assertEqual(state.velocities.tobytes(order="C"), input_velocity_bytes)

    def test_all_far_trigonometric_kepler_branch_matches_public_wh(self):
        state, plan, spec = hybrid_fixture(steps=1, fixed_step=0.6)
        original_node = wh_runtime._node_guard_verdict
        original_path = wh_runtime._path_guard_verdict

        def series_independent_node(**kwargs):
            kwargs["spec"] = dataclasses.replace(
                kwargs["spec"], fixed_step=0.125
            )
            return original_node(**kwargs)

        def series_independent_path(**kwargs):
            kwargs["spec"] = dataclasses.replace(
                kwargs["spec"], fixed_step=0.125
            )
            return original_path(**kwargs)

        # The v1 far-step fraction guard normally keeps the public trajectory
        # inside the series branch.  This controlled guard-only test exercises
        # the shared trigonometric arithmetic without altering the proposer,
        # commit, accounting, replay, or either result projection.
        with mock.patch.object(
            wh_runtime, "_node_guard_verdict", side_effect=series_independent_node
        ), mock.patch.object(
            wh_runtime, "_path_guard_verdict", side_effect=series_independent_path
        ):
            hybrid = integrate_hybrid_wisdom_holman_rkf78_trajectory(
                state, plan, spec
            )
            wh = integrate_wisdom_holman_trajectory(
                state, plan, spec.wisdom_holman_spec
            )
        self.assertTrue(hybrid.all_far)
        self.assertEqual(
            hybrid.final_positions.tobytes(order="C"),
            wh.final_positions.tobytes(order="C"),
        )
        self.assertEqual(
            hybrid.final_velocities.tobytes(order="C"),
            wh.final_velocities.tobytes(order="C"),
        )
        self.assertEqual(
            hybrid.primary_counts.accepted_wh_work.universal_series_terms_evaluated,
            0,
        )
        self.assertEqual(
            hybrid.wisdom_holman_result_content_sha256,
            wh.result_content_sha256,
        )

    def test_far_near_far_is_memoryless_and_rebinds_exactly_once(self):
        state, plan, spec = hybrid_fixture(
            steps=3, checkpoints=(0, 1, 2, 3), atol=1.0e-5
        )
        original = wh_runtime._path_guard_verdict
        calls = 0

        def controlled_path(**kwargs):
            nonlocal calls
            calls += 1
            if calls in (2, 5):
                reason = "FINITE_KEPLER_DRIFT_CLEARANCE_UNCERTIFIED_BY_FAR_SCREEN"
                return wh_runtime._PathGuardVerdict(
                    None,
                    reason,
                    ((0, 1),),
                    TrajectoryDomainError("controlled finite path screen"),
                )
            return original(**kwargs)

        with mock.patch.object(
            wh_runtime, "_path_guard_verdict", side_effect=controlled_path
        ):
            result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
                state, plan, spec
            )
        self.assertEqual(
            tuple(record.mode for record in result.outer_step_records),
            (
                "WISDOM_HOLMAN_FAR",
                "CARTESIAN_RKF78_NEAR_FULL_INTERVAL",
                "WISDOM_HOLMAN_FAR",
            ),
        )
        work = tuple(record.decision.work for record in result.outer_step_records)
        self.assertEqual(
            tuple(item.cartesian_to_jacobi_transforms_completed for item in work),
            (1, 0, 1),
        )
        self.assertEqual(
            tuple(item.force_evaluations_completed for item in work),
            (2, 0, 2),
        )
        self.assertEqual(result.primary_counts.far_accepted_count, 2)
        self.assertEqual(result.primary_counts.far_discarded_count, 1)
        self.assertEqual(result.primary_counts.private_encounter_execution_count, 1)
        self.assertEqual(result.primary_counts.discarded_wh_work.path_guard_evaluations, 1)

    def test_backward_near_uses_signed_outer_duration_not_label_delta(self):
        state, plan, spec = hybrid_fixture(
            steps=1,
            fixed_step=-0.125,
            wh_floor=0.9,
            periapse_floor=0.5,
        )
        result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
            state, plan, spec
        )
        child = result.outer_step_records[0].private_encounter
        self.assertIsNotNone(child)
        assert child is not None
        self.assertEqual(child.segment_spec.duration.hex(), (-0.125).hex())
        self.assertTrue(all(step < 0.0 for step in child.accepted_signed_steps))
        self.assertEqual(child.segment_spec.endpoint_epoch, result.outer_epochs[1])

    def test_initial_contact_is_fatal_and_returns_no_partial_result(self):
        state, plan, spec = hybrid_fixture(
            steps=1,
            wh_floor=0.9,
            periapse_floor=0.5,
            child_floor=1.0,
        )
        with self.assertRaises(encounter_runtime.EncounterError):
            integrate_hybrid_wisdom_holman_rkf78_trajectory(state, plan, spec)

    def test_candidate_force_fatal_never_enters_near_or_returns_partial_state(self):
        state, plan, spec = hybrid_fixture(steps=1)
        original = wh_runtime.evaluate_force_plan
        force_calls = 0
        child_calls = 0

        def failing_force(snapshot, force_plan):
            nonlocal force_calls
            force_calls += 1
            if force_calls == 2:
                raise TrajectoryDomainError("arbitrary endpoint force failure")
            return original(snapshot, force_plan)

        def forbidden_child(*args, **kwargs):
            nonlocal child_calls
            child_calls += 1
            raise AssertionError("fatal WH work must not switch to encounter mode")

        with mock.patch.object(
            wh_runtime, "evaluate_force_plan", side_effect=failing_force
        ), mock.patch.object(
            encounter_runtime, "_execute", side_effect=forbidden_child
        ):
            with self.assertRaisesRegex(
                TrajectoryDomainError, "arbitrary endpoint force failure"
            ):
                integrate_hybrid_wisdom_holman_rkf78_trajectory(
                    state, plan, spec
                )
        self.assertEqual(force_calls, 2)
        self.assertEqual(child_calls, 0)

    def test_checkpoint_cadence_and_discarded_work_are_outer_lattice_only(self):
        state, plan, spec = hybrid_fixture(
            steps=2,
            checkpoints=(0, 1, 2),
            wh_floor=0.9,
            periapse_floor=0.5,
        )
        result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
            state, plan, spec
        )
        self.assertEqual(
            tuple(checkpoint.accepted_steps for checkpoint in result.checkpoints),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(checkpoint.rejected_steps for checkpoint in result.checkpoints),
            (0, 0, 0),
        )
        self.assertEqual(
            result.primary_counts.total_probe_work.path_guard_evaluations,
            sum(
                record.decision.work.path_guard_evaluations
                for record in result.outer_step_records
            ),
        )
        self.assertEqual(
            result.total_public_call_counts.outer_records,
            2 * result.primary_counts.outer_records,
        )
        self.assertEqual(
            result.validation_replay_counts.retained_near_digest_bytes,
            result.primary_counts.retained_near_digest_bytes,
        )

    def test_result_arrays_are_owned_readonly_disjoint_and_do_not_alias_input(self):
        state, plan, spec = hybrid_fixture(
            steps=2, checkpoints=(0, 1, 2), wh_floor=0.9, periapse_floor=0.5
        )
        result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
            state, plan, spec
        )
        retained = [result.initial_snapshot.positions, result.initial_snapshot.velocities]
        for checkpoint in result.checkpoints:
            retained.extend((checkpoint.positions, checkpoint.velocities))
        for array in retained:
            self.assertIs(type(array), np.ndarray)
            self.assertEqual(array.dtype, np.dtype(np.float64))
            self.assertTrue(array.flags.owndata)
            self.assertFalse(array.flags.writeable)
            self.assertFalse(np.shares_memory(array, state.positions))
            self.assertFalse(np.shares_memory(array, state.velocities))
        for index, left in enumerate(retained):
            for right in retained[index + 1 :]:
                self.assertFalse(np.shares_memory(left, right))

    def test_coherent_child_digest_mutation_is_rejected_by_full_replay(self):
        state, plan, spec = hybrid_fixture(
            steps=1, wh_floor=0.9, periapse_floor=0.5
        )
        result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
            state, plan, spec
        )
        record = result.outer_step_records[0]
        assert record.private_encounter is not None
        changed_child = dataclasses.replace(
            record.private_encounter,
            final_state_content_sha256="0" * 64,
        )
        changed_record = dataclasses.replace(
            record, private_encounter=changed_child
        )
        records = (changed_record,)
        step_sha = hybrid_runtime._step_ledger_sha256(records)
        changed_run = dataclasses.replace(
            hybrid_runtime._run_from_result(result), outer_step_records=records
        )
        result_sha = hybrid_runtime._result_sha256(
            initial_snapshot=result.initial_snapshot,
            force_plan=result.force_plan,
            integration_spec=result.integration_spec,
            coordinate_binding=result.coordinate_binding,
            all_epochs=result.outer_epochs,
            checkpoint_epochs=result.checkpoint_epochs,
            run=changed_run,
            primary_counts=result.primary_counts,
            replay_counts=result.validation_replay_counts,
            total_counts=result.total_public_call_counts,
            schedule_sha256=result.schedule_content_sha256,
            step_ledger_sha256=step_sha,
        )
        with self.assertRaises(HybridContractError):
            dataclasses.replace(
                result,
                outer_step_records=records,
                step_ledger_content_sha256=step_sha,
                result_content_sha256=result_sha,
            )

    def test_exact_public_component_types_reject_subclasses_and_malformed_records(self):
        state, plan, spec = hybrid_fixture(steps=1)
        result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
            state, plan, spec
        )

        class TextSubclass(str):
            pass

        with self.assertRaises(HybridContractError):
            dataclasses.replace(result, backend_id=TextSubclass("numpy"))
        with self.assertRaises(HybridContractError):
            dataclasses.replace(result, snapshot_id=TextSubclass(result.snapshot_id))
        record = result.outer_step_records[0]
        with self.assertRaises(HybridContractError):
            HybridOuterStepRecord(
                outer_step_index=True,
                start_epoch=record.start_epoch,
                endpoint_epoch=record.endpoint_epoch,
                signed_step=record.signed_step,
                mode=record.mode,
                probe_committed=record.probe_committed,
                decision=record.decision,
                private_encounter=record.private_encounter,
            )

    def test_result_and_diagnostics_reject_lengths_before_nested_traversal(self):
        state, plan, spec = hybrid_fixture(steps=1)
        result = integrate_hybrid_wisdom_holman_rkf78_trajectory(
            state, plan, spec
        )
        with mock.patch.object(
            wh_runtime,
            "_validate_checkpoint_schema",
            side_effect=AssertionError("checkpoint traversal preceded length check"),
        ):
            with self.assertRaises(HybridContractError):
                dataclasses.replace(result, checkpoints=(object(),))
        with mock.patch.object(
            wh_runtime,
            "_validate_ledger_schema",
            side_effect=AssertionError("ledger traversal preceded force-scope check"),
        ):
            with self.assertRaises(HybridContractError):
                dataclasses.replace(
                    result,
                    force_model_ids=("force.newtonian.point_mass", "forged"),
                    force_ledger=(object(), object()),
                )
        diagnostics = result.wisdom_holman_diagnostics
        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        with mock.patch.object(
            hybrid_runtime.math,
            "isfinite",
            side_effect=AssertionError("diagnostic traversal preceded tuple cap"),
        ):
            with self.assertRaises(HybridContractError):
                dataclasses.replace(
                    diagnostics,
                    minimum_pair_endpoint_separations=(object(),) * 121,
                )

    def test_public_export_and_dataclass_field_rosters_are_exact(self):
        expected_exports = [
            "HybridContractError",
            "HybridDomainError",
            "HybridError",
            "HybridLaneExecutionCounts",
            "HybridOuterStepRecord",
            "HybridProbeWorkTotals",
            "HybridStepLimitError",
            "HybridWisdomHolmanDiagnostics",
            "HybridWisdomHolmanRKF78Result",
            "integrate_hybrid_wisdom_holman_rkf78_trajectory",
        ]
        self.assertEqual(hybrid_runtime.__all__, expected_exports)
        self.assertEqual(hybrid_runtime.__all__, sorted(hybrid_runtime.__all__))
        expected_fields = {
            HybridProbeWorkTotals: HYBRID_FAR_PROBE_WORK_COUNTER_FIELD_ROSTER,
            HybridLaneExecutionCounts: (
                "outer_records",
                "far_probe_count",
                "far_accepted_count",
                "far_discarded_count",
                "private_encounter_execution_count",
                "near_accepted_substeps",
                "retained_near_digest_bytes",
                "checkpoint_count",
                "accepted_wh_work",
                "discarded_wh_work",
                "total_probe_work",
                "private_encounter_counts",
            ),
            HybridOuterStepRecord: (
                "outer_step_index",
                "start_epoch",
                "endpoint_epoch",
                "signed_step",
                "mode",
                "probe_committed",
                "decision",
                "private_encounter",
            ),
            HybridWisdomHolmanDiagnostics: (
                "maximum_universal_solver_iterations",
                "maximum_universal_solver_bracket_expansions",
                "maximum_kepler_time_residual",
                "maximum_kepler_residual_tolerance",
                "maximum_kepler_lagrange_identity_error",
                "maximum_kepler_energy_error",
                "maximum_kepler_angular_momentum_error",
                "maximum_translation_force_residual",
                "minimum_jacobi_periapses",
                "maximum_jacobi_eccentricities",
                "maximum_interaction_force_ratios",
                "maximum_orbit_step_fractions",
                "maximum_periapse_step_fractions",
                "minimum_pair_endpoint_separations",
                "minimum_pair_path_lower_bounds",
                "minimum_pair_clearance_after_margins",
                "minimum_secondary_hill_floor_ratios",
                "maximum_barycenter_position_norm",
                "maximum_barycenter_velocity_norm",
                "initial_barycenter_position_cap",
                "initial_barycenter_velocity_cap",
            ),
            HybridWisdomHolmanRKF78Result: (
                "snapshot_id",
                "plan_id",
                "backend_id",
                "device",
                "dtype",
                "backend_spec",
                "initial_snapshot",
                "force_plan",
                "integration_spec",
                "coordinate_binding",
                "outer_epochs",
                "checkpoint_step_indices",
                "checkpoint_epochs",
                "checkpoints",
                "outer_step_records",
                "force_model_ids",
                "force_ledger",
                "primary_counts",
                "validation_replay_counts",
                "total_public_call_counts",
                "wisdom_holman_diagnostics",
                "wisdom_holman_schedule_content_sha256",
                "wisdom_holman_result_content_sha256",
                "schedule_content_sha256",
                "step_ledger_content_sha256",
                "result_content_sha256",
                "method_id",
                "validation_replay_policy",
                "validation_replay_count",
                "schedule_checksum_algorithm",
                "schedule_checksum_domain",
                "step_ledger_checksum_algorithm",
                "step_ledger_checksum_domain",
                "private_encounter_checksum_algorithm",
                "result_content_checksum_algorithm",
                "result_content_checksum_domain",
                "claim_scope",
                "nonclaims",
                "outer_step_adaptive",
                "encounter_substeps_adaptive",
                "symplectic",
                "time_reversible",
                "dense_output",
                "event_detection",
                "collision_detection",
                "collision_response",
                "regularized",
                "global_clearance_claimed",
                "global_order_claimed",
                "superiority_claimed",
                "evidence_class",
                "registry_authorized",
                "qualification_authorized",
            ),
        }
        for public_type, roster in expected_fields.items():
            self.assertEqual(
                tuple(field.name for field in dataclasses.fields(public_type)),
                roster,
            )

    def test_no_rebound_import_is_required(self):
        root = Path(__file__).resolve().parents[1]
        probe = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                (
                    "import sys; "
                    f"sys.path.insert(0, {str(root / 'src')!r}); "
                    "import jxplanetx.engine.hybrid; "
                    "assert 'rebound' not in sys.modules"
                ),
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertIs(type(hybrid_runtime.__all__), list)
        self.assertIn(
            "integrate_hybrid_wisdom_holman_rkf78_trajectory",
            hybrid_runtime.__all__,
        )


if __name__ == "__main__":
    unittest.main()
