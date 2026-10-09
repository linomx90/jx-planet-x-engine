"""Focused tests for the unpublished DE440 resolved-eleven engine adapter."""

from __future__ import annotations

import dataclasses
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import inspect
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from jxplanetx import solar_system
from jxplanetx.engine.evaluator import evaluate_force_plan
from jxplanetx.solar_system import cspice_execution as execution
from jxplanetx.solar_system import cspice_execution_contracts as evidence
from jxplanetx.solar_system import engine_preparation as runtime
from jxplanetx.solar_system import engine_preparation_contracts as contracts
from jxplanetx.solar_system.contracts import (
    SolarSystemContractError,
    SolarSystemDataError,
)
from jxplanetx.solar_system.cspice_execution_contracts import (
    CspiceSpkgeoExecutionReceipt,
)
from jxplanetx.solar_system.engine_preparation import (
    PreparedDe440NewtonianInputs,
    prepare_de440_newtonian_engine_inputs,
    validate_prepared_de440_newtonian_inputs,
)
from jxplanetx.solar_system.engine_preparation_contracts import (
    De440GmProjection,
    De440NewtonianInitialState,
    De440NewtonianPreparationReceipt,
    De440ResolvedEarthMoonNewtonianSpec,
    TdbEngineEpochBinding,
    validate_de440_gm_projection,
    validate_de440_newtonian_initial_state,
    validate_de440_newtonian_preparation_receipt,
    validate_de440_resolved_earth_moon_newtonian_spec,
    validate_tdb_engine_epoch_binding,
)
from jxplanetx.solar_system.serialization import canonical_json, domain_sha256


_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_m4d_cspice_contract_fixtures",
    Path(__file__).with_name("test_solar_system_cspice_execution_contracts.py"),
)
if _FIXTURE_SPEC is None or _FIXTURE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("frozen M4C2A fixture module is unavailable")
fixtures = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(fixtures)

_LIVE_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_m4d_cspice_execution_fixtures",
    Path(__file__).with_name("test_solar_system_cspice_execution.py"),
)
if _LIVE_FIXTURE_SPEC is None or _LIVE_FIXTURE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("frozen M4C2B fixture module is unavailable")
live_fixtures = importlib.util.module_from_spec(_LIVE_FIXTURE_SPEC)
_LIVE_FIXTURE_SPEC.loader.exec_module(live_fixtures)


EXPECTED_CONTRACT_EXPORTS = [
    "De440GmProjection",
    "De440NewtonianInitialState",
    "De440NewtonianPreparationReceipt",
    "De440ResolvedEarthMoonNewtonianSpec",
    "TdbEngineEpochBinding",
    "validate_de440_gm_projection",
    "validate_de440_newtonian_initial_state",
    "validate_de440_newtonian_preparation_receipt",
    "validate_de440_resolved_earth_moon_newtonian_spec",
    "validate_tdb_engine_epoch_binding",
]
EXPECTED_RUNTIME_EXPORTS = [
    "PreparedDe440NewtonianInputs",
    "prepare_de440_newtonian_engine_inputs",
    "validate_prepared_de440_newtonian_inputs",
]

PROJECTION_FIELDS = (
    "naif_id",
    "kernel_keyword",
    "kernel_value_lexeme",
    "source_gravitational_parameter",
    "si_conversion",
    "target_gravitational_parameter",
    "projection_policy",
    "content_integrity_class",
    "content_sha256",
)
EPOCH_FIELDS = (
    "binding_id",
    "effective_epoch",
    "effective_binary64_et",
    "engine_time_scale",
    "engine_time_unit_id",
    "engine_epoch_at_origin",
    "mapping_policy",
    "future_mapping_status",
    "content_integrity_class",
    "content_sha256",
)
SPEC_FIELDS = (
    "spec_id",
    "constants_artifact",
    "constants_license_artifact",
    "ordered_gm_projections",
    "ordered_body_parameters",
    "source_frame",
    "target_frame",
    "source_unit_system",
    "target_unit_system",
    "roster_policy",
    "origin_policy",
    "force_scope",
    "integrator_scope",
    "mass_policy",
    "radius_policy",
    "engine_alias_policy",
    "omitted_physics",
    "evidence_class",
    "registry_authorized",
    "qualification_authorized",
    "content_integrity_class",
    "content_sha256",
)
PREPARATION_FIELDS = (
    "receipt_id",
    "source_execution",
    "model_spec",
    "selected_lane_policy",
    "source_state_payload_byte_length",
    "source_state_payload_sha256",
    "converted_ssb_payload_sha256",
    "state_conversion_policy",
    "centroid_position_exact_pairs",
    "centroid_velocity_exact_pairs",
    "recenter_policy",
    "normalized_position_residual_exact_pairs",
    "normalized_velocity_residual_exact_pairs",
    "recentered_payload_sha256",
    "gm_payload_sha256",
    "epoch_binding",
    "gm_artifact_verification",
    "gm_license_verification",
    "parameter_read_status",
    "primary_preparation_semantic_sha256",
    "replay_preparation_semantic_sha256",
    "semantic_replay_scope",
    "semantic_replay_status",
    "preparation_status",
    "evidence_class",
    "content_integrity_class",
    "semantic_content_sha256",
    "content_sha256",
)
STATE_FIELDS = (
    "state_id",
    "force_plan_id",
    "preparation_receipt",
    "body_ids",
    "naif_ids",
    "component_order",
    "state_shape",
    "positions",
    "velocities",
    "gravitational_parameters",
    "masses",
    "radii",
    "massive",
    "engine_epoch",
    "engine_time_scale",
    "engine_frame",
    "engine_origin",
    "engine_axes",
    "engine_length_unit",
    "engine_time_unit",
    "engine_mass_unit",
    "engine_unit_system_id",
    "state_stage",
    "evidence_class",
    "registry_authorized",
    "qualification_authorized",
    "semantic_content_sha256",
    "content_integrity_class",
    "content_sha256",
)

EXPECTED_GM_PAYLOAD_SHA256 = (
    "585e6e6c1c59f28fc2d8ca02ee4d196318f22f90d8430a6b16265e013896f0cb"
)
EXPECTED_LIVE_SOURCE_SHA256 = "d71de197fb9d3d693be81929bb547939b19e0ed41ed455b2168ba910e2931d24"
EXPECTED_LIVE_CONVERTED_SHA256 = "51864f06239cc4feb165af0aac510d14c9b50aead0ea307890f42431291643b5"
EXPECTED_LIVE_RECENTERED_SHA256 = "9103461ff23e8b74aa66c116f59e6df48e5fd4bced1ddf7c455662267e901e91"
EXPECTED_LIVE_COMBINED_SHA256 = "2c22b4823ca84f8f401ffffca7798eacd36b6fe5344010b57eeabb9c166ccf4b"
EXPECTED_LIVE_ACCELERATION_SHA256 = "a0d4036aab9586084615946f412cc751418ee6ab08902af0eb9014eafc9d2a41"
EXPECTED_LIVE_CENTROID_POSITION = (
    (-769981719135617666206633769094125, 7134505395521052820000997376),
    (-13478003447448821254991790064870067, 285380215820842112800039895040),
    (105212162935772619120303753705581, 17836263488802632050002493440),
)
EXPECTED_LIVE_CENTROID_VELOCITY = (
    (55612462343649185303620157251, 12468451882689805802975609707560960),
    (205193601114537370757494045591, 16624602510253074403967479610081280),
    (-707495543196390388608183150533, 49873807530759223211902438830243840),
)
EXPECTED_LIVE_POSITION_RESIDUAL = (
    (-100078745066463149993, 8918131744401316025001246720),
    (-5060906747758014431387, 142690107910421056400019947520),
    (4361003941390639899, 660602351437134520370462720),
)
EXPECTED_LIVE_VELOCITY_RESIDUAL = (
    (-7235778650701792183, 24936903765379611605951219415121920),
    (112698674359434877207, 149621422592277669635707316490731520),
    (-162312186234534479, 1246845188268980580297560970756096),
)
EXPECTED_GM_HEX = tuple(str(row[7]) for row in contracts._BODY_ROWS)
RAW_LEXEMES = tuple(str(row[4]) for row in contracts._BODY_ROWS)
EXPECTED_PROJECTION_SEALS = (
    "8d2cfa60020a6deed20b06400548b552cc10e108029156b51db22552fa298219",
    "e8464dc47a0af8e0030e0fc5bd7240742ba8e8e1db28b024aa5f4352247f1864",
    "20443822debef880ca30168c40b4e575462b42c701175cefb0e2054b128bdb0f",
    "2d7dd8cd69b952e0d5e8087f068b3666e28149bb6ac176ae304dc7d3111ea6be",
    "6d4ea414ec56a52e625a8f5b0afcd576dce3e0734e1a1cd93b2c9a4649096707",
    "de345c388ec343e7198a74ba2a11974aa543f3d6a6d2258c42e4042027b1e26f",
    "3fa2dd105ce9348c59b1cdde0c3ad9dd071b1cdf7b25a1ae4ab4d5506e3c616b",
    "09f64b9274f697ffe4e90c4a0ad3cb7b16367ddc3ee41511a887f40f9ad944d0",
    "24f250b70809d9422541a4e63212ea496a67b68e9cd68834320b775deb8b8250",
    "2c0311d4819194695c870609808ebc09a0316b3c7a8e34b70c4799cf2d207a23",
    "396c39117a741d77465e48e989d81782190152d82eca38e78429cc9ed472da6b",
)
EXPECTED_RECORD_KATS = {
    "projection": (1_029, EXPECTED_PROJECTION_SEALS[0]),
    "epoch": (894, "1da8b5bbade669caf1f605ed61b099b47a934abe0a22c9e72e797a664330602e"),
    "spec": (6_312, "ada7839e975752681382b110ddfdb07a631a9faf278ab830d0f851fef569bc9a"),
    "preparation": (3_668, "0bac57da1aa2b49a7b28d753d49498c33eafde90e8138a8e0353c9a506d8f4c1"),
    "preparation_semantic": (3_411, "6820b644bb2130e007cb4b967fa170b368817e52ff42454014b80d5ecdd63848"),
    "state": (5_166, "5913012efcb1267d630acb1bbb624f9c2016fa587a1ab1ab11bdbee22f525665"),
    "state_semantic": (4_897, "1f46427ea302ef3d12bde01f3fd272c89daa7940864f25b4ee48fbcd55c9b2ad"),
}


def _bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _domain_preimage(domain: str, schema: str, payload: object) -> bytes:
    return domain.encode("utf-8") + b"\x00" + schema.encode("utf-8") + b"\x00" + canonical_json(payload)


def _synthetic_worker_result(
    abi: str,
    targets: tuple[int, ...],
    *,
    zero_source_sign: float | None = None,
) -> dict[str, object]:
    result = live_fixtures.worker_result(abi, targets=targets)
    for index, row in enumerate(result["targets"]):
        base = float(index + 1)
        state = (
            zero_source_sign if zero_source_sign is not None and index == 0 else base * 1_000_000.0 + 1.0,
            base * 2_000_000.0 + 2.0,
            base * 3_000_000.0 + 3.0,
            base * 0.01,
            base * -0.02,
            base * 0.03,
        )
        state_hex = [_bits(value).hex() for value in state]
        row["spkgeo_state_be_hex"] = state_hex
        row["spkgeo_light_time_be_hex"] = _bits(base).hex()
        leg = row["legs"][0]
        leg["begin_daf_address"] = 1_000 + index * 1_000
        leg["end_daf_address"] = 1_099 + index * 1_000
        leg["spkpvn_state_be_hex"] = list(state_hex)
    return result


def source_receipt(
    abi: str = "cp314-cp314",
    *,
    whole: int = 0,
    fraction: float = 0.0,
    zero_source_sign: float | None = None,
) -> CspiceSpkgeoExecutionReceipt:
    projection = live_fixtures.projection_at(
        abi,
        targets=contracts._BODY_NAIF_IDS,
        whole=whole,
        fraction=fraction,
    )
    provider = projection.effective_query.provider
    by_id = {item.artifact_id: item for item in provider.artifacts}
    result = _synthetic_worker_result(
        abi,
        contracts._BODY_NAIF_IDS,
        zero_source_sign=zero_source_sign,
    )
    implementation_ids = provider.identity.ordered_implementation_artifact_ids

    def one_lane(role: str, nonce: str, batch_id: str, seed: int):
        implementation = tuple(
            fixtures.local_receipt(by_id[artifact_id], seed + index)
            for index, artifact_id in enumerate(implementation_ids)
        )
        return execution._build_lane(
            role=role,
            nonce=nonce,
            batch_id=batch_id,
            projection=projection,
            result=result,
            implementation_receipts=implementation,
            pre_spk_receipt=fixtures.local_receipt(by_id[fixtures.SPK_ID], seed + 100),
            post_spk_receipt=fixtures.local_receipt(by_id[fixtures.SPK_ID], seed + 101),
            worker_sha256=execution._EXPECTED_WORKER_SHA256,
        )

    return CspiceSpkgeoExecutionReceipt(
        receipt_id="receipt.cspice.execution.resolved11.fixture",
        primary_lane=one_lane("PRIMARY", "a" * 64, "batch.resolved11.primary", 100),
        replay_lane=one_lane("REPLAY", "b" * 64, "batch.resolved11.replay", 2_000),
        semantic_replay_comparison_scope=evidence._REPLAY_COMPARISON_SCOPE,
        semantic_replay_status=evidence._REPLAY_STATUS,
        claim_scope=evidence._REPLAY_CLAIM_SCOPE,
        content_integrity_class=contracts._CONTENT_INTEGRITY_CLASS,
    )


def _mocked_prepare(source: CspiceSpkgeoExecutionReceipt | None = None):
    if source is None:
        source = source_receipt()
    gm = contracts._artifact_binding("CONSTANTS")
    rules = contracts._artifact_binding("LICENSE")
    receipts = {
        gm.artifact_id: fixtures.local_receipt(gm, 31_000),
        rules.artifact_id: fixtures.local_receipt(rules, 32_000),
    }

    def verify(artifact, _root_fd):
        return receipts[artifact.artifact_id]

    def read(_root_fd, artifact, _parser):
        if artifact.artifact_id == gm.artifact_id:
            return RAW_LEXEMES
        return "EXACT_RAW_NAIF_RULES_BYTES_OBSERVED_NONAUTHORIZING"

    with tempfile.TemporaryDirectory() as directory:
        root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with (
                mock.patch.object(runtime, "verify_local_artifact_bytes", side_effect=verify),
                mock.patch.object(runtime, "_read_held_artifact", side_effect=read),
            ):
                value = prepare_de440_newtonian_engine_inputs(
                    source,
                    parameter_root_directory_fd=root_fd,
                    receipt_id="receipt.de440.preparation.fixture",
                    state_id="state.de440.resolved11.fixture",
                    plan_id="plan.de440.resolved11.fixture",
                )
        finally:
            os.close(root_fd)
    return value


def _decimal_newtonian_oracle(positions, gravitational_parameters):
    """Independent high-precision direct all-pairs Newtonian acceleration."""

    rows: list[tuple[float, float, float]] = []
    with localcontext() as context:
        context.prec = 100
        decimal_positions = tuple(
            tuple(Decimal.from_float(float(component)) for component in row)
            for row in positions
        )
        decimal_gm = tuple(
            Decimal.from_float(float(value)) for value in gravitational_parameters
        )
        for target_index, target in enumerate(decimal_positions):
            acceleration = [Decimal(0), Decimal(0), Decimal(0)]
            for source_index, source in enumerate(decimal_positions):
                if source_index == target_index:
                    continue
                displacement = tuple(
                    source[component] - target[component] for component in range(3)
                )
                radius_squared = sum(
                    (component * component for component in displacement), Decimal(0)
                )
                denominator = radius_squared * context.sqrt(radius_squared)
                for component in range(3):
                    acceleration[component] += (
                        decimal_gm[source_index] * displacement[component] / denominator
                    )
            rows.append(tuple(float(component) for component in acceleration))
    return tuple(rows)


def _live_prepared_from_environment() -> PreparedDe440NewtonianInputs:
    artifact_root = os.environ.get("JXPLANETX_M4C2B_ARTIFACT_ROOT")
    parameter_root = os.environ.get("JXPLANETX_M4D_PARAMETER_ROOT")
    if not artifact_root or not parameter_root:
        raise unittest.SkipTest(
            "JXPLANETX_M4C2B_ARTIFACT_ROOT and JXPLANETX_M4D_PARAMETER_ROOT are not configured"
        )
    if sys.version_info[:2] == (3, 12):
        abi = "cp312-cp312"
    elif sys.version_info[:2] == (3, 14):
        abi = "cp314-cp314"
    else:
        raise unittest.SkipTest("live profile supports only retained CPython 3.12/3.14")
    projection = live_fixtures.projection_at(
        abi,
        targets=contracts._BODY_NAIF_IDS,
        whole=0,
        fraction=0.0,
    )
    artifact_fd = os.open(artifact_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        source = execution.execute_cspice_spkgeo_replay(
            projection,
            artifact_root_directory_fd=artifact_fd,
            receipt_id="receipt.m4d.live.adapter",
            primary_batch_id="batch.m4d.live.adapter.primary",
            replay_batch_id="batch.m4d.live.adapter.replay",
            primary_execution_instance_nonce="c" * 64,
            replay_execution_instance_nonce="d" * 64,
        )
    finally:
        os.close(artifact_fd)
    parameter_fd = os.open(parameter_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return prepare_de440_newtonian_engine_inputs(
            source,
            parameter_root_directory_fd=parameter_fd,
            receipt_id="receipt.m4d.live.preparation",
            state_id="state.m4d.live.initial",
            plan_id="plan.m4d.live.force",
        )
    finally:
        os.close(parameter_fd)


class SchemaAndProjectionTests(unittest.TestCase):
    def test_exact_exports_fields_and_signatures(self) -> None:
        self.assertEqual(contracts.__all__, sorted(EXPECTED_CONTRACT_EXPORTS))
        self.assertEqual(runtime.__all__, sorted(EXPECTED_RUNTIME_EXPORTS))
        for record, expected in (
            (De440GmProjection, PROJECTION_FIELDS),
            (TdbEngineEpochBinding, EPOCH_FIELDS),
            (De440ResolvedEarthMoonNewtonianSpec, SPEC_FIELDS),
            (De440NewtonianPreparationReceipt, PREPARATION_FIELDS),
            (De440NewtonianInitialState, STATE_FIELDS),
        ):
            self.assertEqual(tuple(field.name for field in dataclasses.fields(record)), expected)
            self.assertTrue(record.__dataclass_params__.frozen)
            self.assertEqual(record.__dataclass_params__.eq, False)
            self.assertEqual(record.__slots__, expected)
        self.assertEqual(
            str(inspect.signature(prepare_de440_newtonian_engine_inputs)),
            "(source_execution: 'CspiceSpkgeoExecutionReceipt', *, parameter_root_directory_fd: 'int', receipt_id: 'str', state_id: 'str', plan_id: 'str') -> 'PreparedDe440NewtonianInputs'",
        )

    def test_strict_gm_parser_and_projection_kat(self) -> None:
        source = "\n".join(
            f"   {row[3]} = ( {row[4]} )" for row in contracts._BODY_ROWS
        ).encode("ascii")
        self.assertEqual(runtime._parse_gm_bytes(source), RAW_LEXEMES)
        projections = runtime._build_projections(RAW_LEXEMES)
        self.assertEqual(tuple(item.naif_id for item in projections), contracts._BODY_NAIF_IDS)
        self.assertEqual(
            tuple(item.target_gravitational_parameter.operational_value.hex() for item in projections),
            EXPECTED_GM_HEX,
        )
        for projection in projections:
            validate_de440_gm_projection(projection)
            self.assertEqual(projection.source_gravitational_parameter.classification, "ESTIMATED")
            self.assertNotIn("E", projection.source_gravitational_parameter.source_decimal)
            self.assertNotIn("D", projection.source_gravitational_parameter.source_decimal)
        payload = b"".join(
            struct.pack(">d", item.target_gravitational_parameter.operational_value)
            for item in projections
        )
        self.assertEqual(len(payload), 88)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), EXPECTED_GM_PAYLOAD_SHA256)

    def test_parser_rejects_duplicate_mutation_and_non_ascii(self) -> None:
        exact = "\n".join(
            f"   {row[3]} = ( {row[4]} )" for row in contracts._BODY_ROWS
        )
        with self.assertRaises(SolarSystemDataError):
            runtime._parse_gm_bytes((exact + f"\n   BODY10_GM = ( {RAW_LEXEMES[0]} )").encode())
        with self.assertRaises(SolarSystemDataError):
            runtime._parse_gm_bytes(exact.replace(RAW_LEXEMES[0], "1.0E+00").encode())
        with self.assertRaises(SolarSystemDataError):
            runtime._parse_gm_bytes(exact.encode() + b"\xff")
        with self.assertRaises(SolarSystemDataError):
            runtime._parse_rules_bytes(b"not the exact retained rules")

    def test_text_caps_precede_strip_and_child_schemas_are_exact(self) -> None:
        with mock.patch.object(contracts, "_m1_ref", side_effect=AssertionError("traversed")):
            with self.assertRaises(SolarSystemContractError):
                TdbEngineEpochBinding(
                    binding_id="x" * (contracts._MAXIMUM_TEXT_CODEPOINTS + 1),
                    effective_epoch=object(),
                    effective_binary64_et=0.0,
                    engine_time_scale="TDB",
                    engine_time_unit_id="x",
                    engine_epoch_at_origin=0.0,
                    mapping_policy="x",
                    future_mapping_status="x",
                    content_integrity_class=contracts._CONTENT_INTEGRITY_CLASS,
                )
        self.assertEqual(
            contracts._M1_SCHEMA_BY_TYPE[contracts.PhysicalConstant],
            "jxplanetx.solar-system.physical-constant.payload.v1",
        )
        self.assertEqual(
            contracts._M1_SCHEMA_BY_TYPE[contracts.CoordinateEpoch],
            "jxplanetx.solar-system.coordinate-epoch.payload.v1",
        )


class EndToEndPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = source_receipt()
        cls.prepared = _mocked_prepare(cls.source)

    def test_complete_wrapper_and_frozen_force_evaluator(self) -> None:
        value = self.prepared
        validate_prepared_de440_newtonian_inputs(value)
        result = evaluate_force_plan(value.snapshot, value.force_plan)
        self.assertEqual(result.acceleration.shape, (11, 3))
        self.assertTrue(np.isfinite(result.acceleration).all())
        self.assertEqual(len(result.contributions), 1)
        self.assertEqual(value.force_plan.backend.tile_size, 1)
        self.assertFalse(value.force_plan.backend.allow_fallback)
        self.assertFalse(value.force_plan.backend.fast_math)

    def test_epoch_alias_and_nonqualification_scope(self) -> None:
        value = self.prepared
        binding = value.preparation_receipt.epoch_binding
        self.assertEqual(value.snapshot.epoch.hex(), "0x0.0p+0")
        self.assertEqual(binding.engine_epoch_at_origin.hex(), "0x0.0p+0")
        self.assertEqual(binding.effective_binary64_et.hex(), "0x0.0p+0")
        self.assertEqual(value.snapshot.time_scale, "TDB")
        self.assertEqual(value.snapshot.frame, "BARYCENTRIC_INERTIAL")
        self.assertEqual(value.snapshot.origin, "BARYCENTER")
        self.assertEqual(value.snapshot.axes, "CARTESIAN_RIGHT_HANDED")
        spec = value.preparation_receipt.model_spec
        self.assertIn("NOT_WISDOM_HOLMAN_OR_ORDERED_JACOBI_QUALIFIED", spec.integrator_scope)
        self.assertEqual(spec.target_frame.origin_kind, "NEWTONIAN_MODEL_BARYCENTER")
        self.assertNotEqual(spec.target_frame.origin_kind, "SOLAR_SYSTEM_BARYCENTER")
        self.assertFalse(spec.registry_authorized)
        self.assertFalse(spec.qualification_authorized)

    def test_nonzero_effective_et_remains_only_in_affine_origin_binding(self) -> None:
        value = _mocked_prepare(source_receipt(whole=1, fraction=0.0))
        binding = value.preparation_receipt.epoch_binding
        self.assertEqual(binding.effective_binary64_et.hex(), "0x1.0000000000000p+0")
        self.assertEqual(binding.effective_epoch.whole, 1)
        self.assertEqual(value.initial_state.engine_epoch.hex(), "0x0.0p+0")
        self.assertEqual(value.snapshot.epoch.hex(), "0x0.0p+0")
        self.assertEqual(
            value.force_plan.models[0].parameter_metadata[0].validity_start.hex(),
            "0x0.0p+0",
        )

    def test_roster_placeholder_and_owned_array_contract(self) -> None:
        value = self.prepared
        self.assertEqual(value.initial_state.naif_ids, (10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9))
        self.assertEqual(
            tuple(body.body_role for body in value.preparation_receipt.model_spec.ordered_body_parameters),
            ("MASSIVE_BODY_CENTER",) * 5 + ("PLANETARY_SYSTEM_BARYCENTER",) * 6,
        )
        arrays = (
            value.snapshot.positions,
            value.snapshot.velocities,
            value.snapshot.gravitational_parameters,
            value.snapshot.masses,
            value.snapshot.radii,
            value.snapshot.massive,
        )
        for array in arrays:
            self.assertTrue(array.flags.owndata)
            self.assertIsNone(array.base)
            self.assertTrue(array.flags.c_contiguous)
            self.assertFalse(array.flags.writeable)
        for index, left in enumerate(arrays):
            for right in arrays[index + 1 :]:
                self.assertFalse(np.shares_memory(left, right))
        self.assertEqual(tuple(value.snapshot.masses), (0.0,) * 11)
        self.assertEqual(tuple(value.snapshot.radii), (0.0,) * 11)
        self.assertTrue(all(math.copysign(1.0, item) > 0 for item in value.snapshot.masses))
        self.assertTrue(all(math.copysign(1.0, item) > 0 for item in value.snapshot.radii))
        self.assertEqual(tuple(value.snapshot.massive), (True,) * 11)

    def test_semantic_replay_and_exact_centroid_residuals(self) -> None:
        receipt = self.prepared.preparation_receipt
        self.assertEqual(
            receipt.primary_preparation_semantic_sha256,
            receipt.replay_preparation_semantic_sha256,
        )
        self.assertIn("SAME_CALLER_NO_PROCESS_INDEPENDENCE_PROOF", receipt.semantic_replay_scope)
        self.assertEqual(receipt.gm_payload_sha256, EXPECTED_GM_PAYLOAD_SHA256)
        self.assertEqual(receipt.source_state_payload_byte_length, 528)
        self.assertEqual(len(receipt.centroid_position_exact_pairs), 3)
        self.assertEqual(len(receipt.normalized_position_residual_exact_pairs), 3)
        self.assertNotEqual(receipt.converted_ssb_payload_sha256, receipt.recentered_payload_sha256)

    def test_literal_record_and_semantic_preimages_are_locked(self) -> None:
        receipt = self.prepared.preparation_receipt
        projection = receipt.model_spec.ordered_gm_projections[0]
        state = self.prepared.initial_state
        cases = (
            (
                "projection",
                contracts._PROJECTION_DOMAIN,
                contracts._PROJECTION_SCHEMA,
                contracts._projection_payload(projection),
                projection.content_sha256,
            ),
            (
                "epoch",
                contracts._EPOCH_DOMAIN,
                contracts._EPOCH_SCHEMA,
                contracts._epoch_payload(receipt.epoch_binding),
                receipt.epoch_binding.content_sha256,
            ),
            (
                "spec",
                contracts._SPEC_DOMAIN,
                contracts._SPEC_SCHEMA,
                contracts._spec_payload(receipt.model_spec),
                receipt.model_spec.content_sha256,
            ),
            (
                "preparation",
                contracts._PREPARATION_DOMAIN,
                contracts._PREPARATION_SCHEMA,
                contracts._preparation_payload(receipt),
                receipt.content_sha256,
            ),
            (
                "preparation_semantic",
                contracts._PREPARATION_SEMANTIC_DOMAIN,
                contracts._PREPARATION_SEMANTIC_SCHEMA,
                contracts._preparation_semantic_payload(receipt),
                receipt.semantic_content_sha256,
            ),
            (
                "state",
                contracts._STATE_DOMAIN,
                contracts._STATE_SCHEMA,
                contracts._initial_state_payload(state),
                state.content_sha256,
            ),
            (
                "state_semantic",
                contracts._STATE_SEMANTIC_DOMAIN,
                contracts._STATE_SEMANTIC_SCHEMA,
                contracts._initial_state_semantic_payload(state),
                state.semantic_content_sha256,
            ),
        )
        for label, domain, schema, payload, observed_seal in cases:
            preimage = _domain_preimage(domain, schema, payload)
            expected_length, expected_sha = EXPECTED_RECORD_KATS[label]
            self.assertEqual(len(preimage), expected_length)
            self.assertEqual(hashlib.sha256(preimage).hexdigest(), expected_sha)
            self.assertEqual(observed_seal, expected_sha)
        self.assertEqual(
            tuple(item.content_sha256 for item in receipt.model_spec.ordered_gm_projections),
            EXPECTED_PROJECTION_SEALS,
        )

    def test_plan_id_is_sealed_and_relabel_cannot_masquerade(self) -> None:
        value = self.prepared
        changed_plan = dataclasses.replace(value.force_plan, plan_id="plan.relabelled")
        forged = PreparedDe440NewtonianInputs(
            value.preparation_receipt,
            value.initial_state,
            value.snapshot,
            changed_plan,
        )
        with self.assertRaises(SolarSystemDataError):
            validate_prepared_de440_newtonian_inputs(forged)
        changed_state = dataclasses.replace(
            value.initial_state,
            force_plan_id="plan.relabelled",
            semantic_content_sha256="",
            content_sha256="",
        )
        validate_de440_newtonian_initial_state(changed_state)
        self.assertNotEqual(changed_state.content_sha256, value.initial_state.content_sha256)
        self.assertEqual(changed_state.semantic_content_sha256, value.initial_state.semantic_content_sha256)

    def test_array_mutation_and_signed_zero_are_detected(self) -> None:
        base = self.prepared
        copied_positions = np.array(base.snapshot.positions, copy=True, order="C")
        copied_positions.flags.writeable = False
        value = PreparedDe440NewtonianInputs(
            base.preparation_receipt,
            base.initial_state,
            dataclasses.replace(base.snapshot, positions=copied_positions),
            base.force_plan,
        )
        array = value.snapshot.positions
        array.flags.writeable = True
        array[0, 0] = -0.0 if array[0, 0] == 0.0 else array[0, 0] + 1.0
        array.flags.writeable = False
        with self.assertRaises(SolarSystemDataError):
            validate_prepared_de440_newtonian_inputs(value)

    def test_true_signed_zero_conversion_and_recenter_rounding_witness(self) -> None:
        self.assertEqual(contracts._convert_state_component(-0.0, velocity=False).hex(), "-0x0.0p+0")
        self.assertEqual(contracts._convert_state_component(-0.0, velocity=True).hex(), "-0x0.0p+0")
        self.assertEqual(
            contracts._round_target_component(Fraction(0, 1), velocity=False).hex(),
            "0x0.0p+0",
        )
        negative = _mocked_prepare(source_receipt(zero_source_sign=-0.0))
        positive_source = source_receipt(zero_source_sign=0.0)
        negative_values = contracts._lane_preparation_values(
            negative.preparation_receipt.source_execution.primary_lane,
            negative.preparation_receipt.model_spec,
        )
        positive_values = contracts._lane_preparation_values(
            positive_source.primary_lane,
            negative.preparation_receipt.model_spec,
        )
        self.assertNotEqual(
            negative_values["converted_payload_sha256"],
            positive_values["converted_payload_sha256"],
        )
        # Exact recentering treats signed zero as the same real coordinate and
        # emits canonical +0 only if the exact post-subtraction result is zero.
        self.assertEqual(
            negative_values["recentered_payload_sha256"],
            positive_values["recentered_payload_sha256"],
        )

    def test_point_validity_metadata_and_evaluator_unit_crosswalk(self) -> None:
        metadata = self.prepared.force_plan.models[0].parameter_metadata[0]
        self.assertEqual(metadata.validity_start.hex(), "0x0.0p+0")
        self.assertEqual(metadata.validity_end.hex(), "0x0.0p+0")
        self.assertEqual(
            metadata.units,
            f"{self.prepared.snapshot.length_unit}^3/{self.prepared.snapshot.time_unit}^2",
        )
        self.assertNotEqual(
            metadata.units,
            contracts._TARGET_GM_UNIT.unit_id,
        )
        self.assertEqual(
            self.prepared.preparation_receipt.model_spec.target_unit_system.gravitational_parameter.unit_id,
            contracts._TARGET_GM_UNIT.unit_id,
        )

    def test_nonzero_kdk_schedule_is_blocked_by_point_validity(self) -> None:
        from jxplanetx.engine.symplectic import integrate_kdk_trajectory
        from jxplanetx.engine.symplectic_contracts import FixedStepKDKSpec
        from jxplanetx.engine.trajectory import TrajectoryContractError

        spec = FixedStepKDKSpec(
            checkpoint_step_indices=(0, 1),
            fixed_step=1.0,
            maximum_steps=1,
            minimum_swept_pair_separation=1.0,
            maximum_pair_frequency_step=1.0,
        )
        with self.assertRaises(TrajectoryContractError):
            integrate_kdk_trajectory(self.prepared.snapshot, self.prepared.force_plan, spec)

    def test_nested_engine_dataclass_tampering_is_revalidated(self) -> None:
        for mutation in ("metadata-list", "models-list", "bool-tile", "model-id"):
            base = self.prepared
            model = dataclasses.replace(base.force_plan.models[0])
            backend = dataclasses.replace(base.force_plan.backend)
            plan = dataclasses.replace(base.force_plan, backend=backend, models=(model,))
            value = PreparedDe440NewtonianInputs(
                base.preparation_receipt,
                base.initial_state,
                base.snapshot,
                plan,
            )
            if mutation == "metadata-list":
                object.__setattr__(model, "parameter_metadata", list(model.parameter_metadata))
            elif mutation == "models-list":
                object.__setattr__(value.force_plan, "models", list(value.force_plan.models))
            elif mutation == "model-id":
                object.__setattr__(model, "MODEL_ID", "force.forged")
            else:
                object.__setattr__(value.force_plan.backend, "tile_size", True)
            with self.assertRaises(SolarSystemDataError):
                validate_prepared_de440_newtonian_inputs(value)

    def test_hostile_engine_fields_fail_before_nested_post_init_traversal(self) -> None:
        base = self.prepared
        cases = []
        huge_snapshot = dataclasses.replace(base.snapshot)
        object.__setattr__(huge_snapshot, "snapshot_id", "x" * 1_000_000)
        cases.append((huge_snapshot, base.force_plan, type(base.snapshot)))
        huge_ids = dataclasses.replace(base.snapshot)
        object.__setattr__(huge_ids, "body_ids", ("X",) * 100_000)
        cases.append((huge_ids, base.force_plan, type(base.snapshot)))

        class FloatSubclass(float):
            pass

        odd_epoch = dataclasses.replace(base.snapshot)
        object.__setattr__(odd_epoch, "epoch", FloatSubclass(0.0))
        cases.append((odd_epoch, base.force_plan, type(base.snapshot)))
        huge_models = dataclasses.replace(base.force_plan)
        object.__setattr__(huge_models, "models", base.force_plan.models * 100_000)
        cases.append((base.snapshot, huge_models, type(base.force_plan)))
        huge_model = dataclasses.replace(base.force_plan.models[0])
        object.__setattr__(huge_model, "MODEL_ID", "x" * 1_000_000)
        huge_model_plan = dataclasses.replace(base.force_plan, models=(huge_model,))
        cases.append((base.snapshot, huge_model_plan, type(huge_model)))
        for snapshot, plan, guarded_type in cases:
            candidate = PreparedDe440NewtonianInputs(
                base.preparation_receipt,
                base.initial_state,
                snapshot,
                plan,
            )
            with mock.patch.object(
                guarded_type,
                "__post_init__",
                side_effect=AssertionError("deep traversal reached"),
            ):
                with self.assertRaises(SolarSystemDataError):
                    validate_prepared_de440_newtonian_inputs(candidate)

    def test_instance_shadowed_post_init_is_never_called(self) -> None:
        base = self.prepared

        def hostile() -> None:
            raise RuntimeError("caller-shadowed method executed")

        snapshot = dataclasses.replace(base.snapshot)
        object.__setattr__(snapshot, "__post_init__", hostile)
        candidate = PreparedDe440NewtonianInputs(
            base.preparation_receipt,
            base.initial_state,
            snapshot,
            base.force_plan,
        )
        validate_prepared_de440_newtonian_inputs(candidate)

    def test_stale_child_refs_and_semantic_mutations_reject(self) -> None:
        receipt = self.prepared.preparation_receipt
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt,
                primary_preparation_semantic_sha256="1" * 64,
                semantic_content_sha256="",
                content_sha256="",
            )
        state = self.prepared.initial_state
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                state,
                preparation_receipt=dataclasses.replace(
                    receipt,
                    receipt_id="receipt.relabelled",
                    semantic_content_sha256="",
                    content_sha256="",
                ),
                semantic_content_sha256=state.semantic_content_sha256,
                content_sha256=state.content_sha256,
            )

    def test_contract_preflight_happens_before_source_traversal(self) -> None:
        with mock.patch.object(runtime, "_require_source_execution", side_effect=AssertionError("traversed")):
            with self.assertRaises(SolarSystemContractError):
                prepare_de440_newtonian_engine_inputs(
                    object(),
                    parameter_root_directory_fd=0,
                    receipt_id="x" * 129,
                    state_id="state",
                    plan_id="plan",
                )


class ImportAndLiteralTests(unittest.TestCase):
    def test_contract_module_has_no_numpy_or_provider_import(self) -> None:
        contract_source = Path(contracts.__file__).read_text(encoding="utf-8")
        runtime_source = Path(runtime.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import numpy", contract_source)
        self.assertNotIn("import spiceypy", contract_source)
        self.assertNotIn("from spiceypy", contract_source)
        self.assertIn("not\nWisdom-Holman", contract_source)
        self.assertIn("not one atomic snapshot", runtime_source)
        self.assertIn("same-UID, root", runtime_source)

    def test_clean_runtime_import_does_not_require_numpy_engine_or_provider(self) -> None:
        source_root = str(Path(__file__).resolve().parents[1] / "src")
        code = f"""
import builtins, sys
sys.path.insert(0, {source_root!r})
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'numpy' or name.startswith('numpy.') or name == 'spiceypy' or name.startswith('spiceypy.'):
        raise ModuleNotFoundError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import jxplanetx.solar_system.engine_preparation as adapter
from jxplanetx.solar_system.contracts import SolarSystemDependencyUnavailableError
assert 'numpy' not in sys.modules
assert 'spiceypy' not in sys.modules
assert 'jxplanetx.engine' not in sys.modules
try:
    adapter._load_engine_contracts()
except SolarSystemDependencyUnavailableError:
    pass
else:
    raise AssertionError('missing NumPy did not map to DependencyUnavailable')
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-S", "-c", code],
            cwd="/",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_held_reader_rejects_symlink_and_truncation_without_fd_or_root_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "constants").mkdir()
            data = b"bounded parameter evidence\n"
            (root / "constants" / "good.tpc").write_bytes(data)
            (root / "constants" / "short.tpc").write_bytes(data[:-1])
            os.symlink("good.tpc", root / "constants" / "link.tpc")
            base = contracts._artifact_binding("CONSTANTS")
            good = dataclasses.replace(
                base,
                logical_locator="constants/good.tpc",
                byte_length=len(data),
                artifact_sha256=hashlib.sha256(data).hexdigest(),
                content_sha256="",
            )
            short = dataclasses.replace(
                good,
                logical_locator="constants/short.tpc",
                content_sha256="",
            )
            link = dataclasses.replace(
                good,
                logical_locator="constants/link.tpc",
                content_sha256="",
            )
            root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            before_fds = len(tuple(Path("/proc/self/fd").iterdir()))
            try:
                self.assertEqual(runtime._read_held_artifact(root_fd, good, bytes), data)
                with self.assertRaises(SolarSystemDataError):
                    runtime._read_held_artifact(root_fd, short, bytes)
                with self.assertRaises((SolarSystemDataError, OSError)):
                    runtime._read_held_artifact(root_fd, link, bytes)
                os.fstat(root_fd)
                self.assertEqual(len(tuple(Path("/proc/self/fd").iterdir())), before_fds)
            finally:
                os.close(root_fd)

    def test_held_reader_detects_preopen_read_and_second_resolution_races(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            constants_dir = root / "constants"
            constants_dir.mkdir()
            selected = constants_dir / "selected.tpc"
            data = b"held-reader-race-evidence\n"
            selected.write_bytes(data)
            base = contracts._artifact_binding("CONSTANTS")
            artifact = dataclasses.replace(
                base,
                logical_locator="constants/selected.tpc",
                byte_length=len(data),
                artifact_sha256=hashlib.sha256(data).hexdigest(),
                content_sha256="",
            )
            root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            baseline_fds = len(tuple(Path("/proc/self/fd").iterdir()))
            try:
                real_open = os.open
                replaced = False

                def replace_before_open(path, flags, *args, **kwargs):
                    nonlocal replaced
                    if path == "selected.tpc" and not replaced:
                        replacement = constants_dir / "replacement.tpc"
                        replacement.write_bytes(data)
                        os.replace(replacement, selected)
                        replaced = True
                    return real_open(path, flags, *args, **kwargs)

                with mock.patch.object(runtime.os, "open", side_effect=replace_before_open):
                    with self.assertRaises(SolarSystemDataError):
                        runtime._read_held_artifact(root_fd, artifact, bytes)
                self.assertEqual(len(tuple(Path("/proc/self/fd").iterdir())), baseline_fds)

                selected.write_bytes(data)
                real_read = os.read
                touched = False

                def mutate_after_read(fd, count):
                    nonlocal touched
                    result = real_read(fd, count)
                    if result and not touched:
                        observed = selected.stat()
                        os.utime(
                            selected,
                            ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000),
                        )
                        touched = True
                    return result

                with mock.patch.object(runtime.os, "read", side_effect=mutate_after_read):
                    with self.assertRaises(SolarSystemDataError):
                        runtime._read_held_artifact(root_fd, artifact, bytes)
                self.assertEqual(len(tuple(Path("/proc/self/fd").iterdir())), baseline_fds)

                selected.write_bytes(data)

                def replace_during_parse(retained):
                    replacement = constants_dir / "second-resolution.tpc"
                    replacement.write_bytes(retained)
                    os.replace(replacement, selected)
                    return retained

                with self.assertRaises(SolarSystemDataError):
                    runtime._read_held_artifact(root_fd, artifact, replace_during_parse)
                os.fstat(root_fd)
                self.assertEqual(len(tuple(Path("/proc/self/fd").iterdir())), baseline_fds)
            finally:
                os.close(root_fd)

    def test_root_package_remains_unpublished_and_serializer_fails_closed(self) -> None:
        self.assertEqual(solar_system.__all__, [])
        projection = runtime._build_projections(RAW_LEXEMES)[0]
        with self.assertRaises(SolarSystemContractError):
            domain_sha256("test.domain", "test.schema", projection)

    def test_external_parameter_root_is_explicitly_opt_in(self) -> None:
        root = os.environ.get("JXPLANETX_M4D_PARAMETER_ROOT")
        if not root:
            self.skipTest("JXPLANETX_M4D_PARAMETER_ROOT is not configured")
        gm = contracts._artifact_binding("CONSTANTS")
        rules = contracts._artifact_binding("LICENSE")
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            gm_receipt = runtime.verify_local_artifact_bytes(gm, root_fd)
            rules_receipt = runtime.verify_local_artifact_bytes(rules, root_fd)
            lexemes = runtime._read_held_artifact(root_fd, gm, runtime._parse_gm_bytes)
            status = runtime._read_held_artifact(root_fd, rules, runtime._parse_rules_bytes)
        finally:
            os.close(root_fd)
        self.assertEqual(lexemes, RAW_LEXEMES)
        self.assertEqual(status, "EXACT_RAW_NAIF_RULES_BYTES_OBSERVED_NONAUTHORIZING")
        gm_receipt.validate_integrity()
        rules_receipt.validate_integrity()


class OptInLiveCalibrationTests(unittest.TestCase):
    def test_exact_m4c2b_parameter_recenter_and_force_kats(self) -> None:
        value = _live_prepared_from_environment()
        receipt = value.preparation_receipt
        self.assertEqual(receipt.source_state_payload_sha256, EXPECTED_LIVE_SOURCE_SHA256)
        self.assertEqual(receipt.converted_ssb_payload_sha256, EXPECTED_LIVE_CONVERTED_SHA256)
        self.assertEqual(receipt.recentered_payload_sha256, EXPECTED_LIVE_RECENTERED_SHA256)
        self.assertEqual(receipt.gm_payload_sha256, EXPECTED_GM_PAYLOAD_SHA256)
        self.assertEqual(receipt.centroid_position_exact_pairs, EXPECTED_LIVE_CENTROID_POSITION)
        self.assertEqual(receipt.centroid_velocity_exact_pairs, EXPECTED_LIVE_CENTROID_VELOCITY)
        self.assertEqual(
            receipt.normalized_position_residual_exact_pairs,
            EXPECTED_LIVE_POSITION_RESIDUAL,
        )
        self.assertEqual(
            receipt.normalized_velocity_residual_exact_pairs,
            EXPECTED_LIVE_VELOCITY_RESIDUAL,
        )
        combined = contracts._state_payload(
            value.initial_state.positions,
            value.initial_state.velocities,
        ) + contracts._gm_payload(value.initial_state.gravitational_parameters)
        self.assertEqual(len(combined), 616)
        self.assertEqual(hashlib.sha256(combined).hexdigest(), EXPECTED_LIVE_COMBINED_SHA256)

        evaluated = evaluate_force_plan(value.snapshot, value.force_plan)
        evaluated_bytes = b"".join(
            struct.pack(">d", float(component))
            for component in evaluated.acceleration.ravel(order="C")
        )
        self.assertEqual(
            hashlib.sha256(evaluated_bytes).hexdigest(),
            EXPECTED_LIVE_ACCELERATION_SHA256,
        )
        oracle = _decimal_newtonian_oracle(
            value.snapshot.positions,
            value.snapshot.gravitational_parameters,
        )
        oracle_bytes = b"".join(
            struct.pack(">d", component) for row in oracle for component in row
        )
        self.assertEqual(
            hashlib.sha256(oracle_bytes).hexdigest(),
            "b557c227c753984e09bcb1d4f4d753033de267ce87c86f343aa70cc3c8511015",
        )
        for target_index, row in enumerate(oracle):
            for component, expected in enumerate(row):
                self.assertTrue(
                    math.isclose(
                        float(evaluated.acceleration[target_index, component]),
                        expected,
                        rel_tol=5e-15,
                        abs_tol=4e-18,
                    )
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
