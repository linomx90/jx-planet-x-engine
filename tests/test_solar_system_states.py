"""Provider-native state declaration tests for Solar-System Milestone 4A."""

from __future__ import annotations

import base64
import dataclasses
from fractions import Fraction
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import unittest
from unittest import mock

from jxplanetx.solar_system import states as states_module
from jxplanetx.solar_system.contracts import (
    EphemerisQuerySpec,
    ExactUnitScale,
    SolarSystemContractError,
    UnitSystemDefinition,
    validate_integrity,
)
from jxplanetx.solar_system.serialization import canonical_json
from jxplanetx.solar_system.states import (
    ProviderNativeStateBatch,
    validate_provider_native_state_batch,
)
from tests.test_solar_system_contracts import (
    SOURCE_ID,
    spk_context,
    synthetic_context,
)


EXPECTED_EXPORTS = [
    "ProviderNativeStateBatch",
    "validate_provider_native_state_batch",
]
EXPECTED_FIELDS = (
    "batch_id",
    "query",
    "state_stage",
    "execution_evidence_status",
    "target_naif_ids",
    "component_order",
    "state_shape",
    "position_unit_id",
    "velocity_length_unit_id",
    "velocity_time_unit_id",
    "velocity_semantics",
    "positions",
    "velocities",
    "target_availability_status",
    "artifact_custody_status",
    "semantic_replay_status",
    "content_integrity_class",
    "content_sha256",
)

STATE_STAGE = (
    "DECLARED_PROVIDER_NATIVE_GEOMETRIC_TARGET_RELATIVE_OBSERVER_STATE_"
    "NO_JX_CONVERSION"
)
EXECUTION_STATUS = "CALLER_SUPPLIED_UNAUTHENTICATED_NO_EXECUTION_RECEIPT"
VELOCITY_SEMANTICS = "COORDINATE_DERIVATIVE_PER_PROVIDER_NATIVE_TIME_UNIT"
ARTIFACT_STATUS = "NO_ARTIFACT_AT_USE_CUSTODY_EVIDENCE"
REPLAY_STATUS = "REQUIRES_PROVIDER_SPECIFIC_SEMANTIC_REPLAY"
INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"

BATCH_DOMAIN = (
    "jxplanetx.solar-system.states.provider-native-state-batch."
    "content-integrity.v1"
)
BATCH_SCHEMA = "ProviderNativeStateBatch.v1"


def make_batch(
    query: EphemerisQuerySpec | None = None,
    *,
    batch_id: str = "batch.fixture.synthetic.v1",
    positions: tuple[float, ...] | None = None,
    velocities: tuple[float, ...] | None = None,
) -> ProviderNativeStateBatch:
    if query is None:
        query = synthetic_context()[1]
    count = len(query.target_naif_ids) * 3
    if positions is None:
        positions = tuple(float(index + 1) for index in range(count))
    if velocities is None:
        velocities = tuple(-float(index + 1) / 8.0 for index in range(count))
    return ProviderNativeStateBatch(
        batch_id=batch_id,
        query=query,
        state_stage=STATE_STAGE,
        execution_evidence_status=EXECUTION_STATUS,
        target_naif_ids=query.target_naif_ids,
        component_order=("X", "Y", "Z"),
        state_shape=(len(query.target_naif_ids), 3),
        position_unit_id=query.output_unit_system.length.unit_id,
        velocity_length_unit_id=query.output_unit_system.length.unit_id,
        velocity_time_unit_id=query.output_unit_system.time.unit_id,
        velocity_semantics=VELOCITY_SEMANTICS,
        positions=positions,
        velocities=velocities,
        target_availability_status=query.target_chain_availability_status,
        artifact_custody_status=ARTIFACT_STATUS,
        semantic_replay_status=REPLAY_STATUS,
        content_integrity_class=INTEGRITY_CLASS,
    )


def tdb_output_units(*, length: str, time: str) -> UnitSystemDefinition:
    if length == "KILOMETRE":
        length_unit = ExactUnitScale(
            "unit.kilometre.defined-exact.v1",
            "LENGTH",
            "si.metre",
            1_000,
            1,
            "DEFINED_EXACT",
            (SOURCE_ID,),
        )
    elif length == "METRE":
        length_unit = ExactUnitScale(
            "unit.metre.defined-exact.v1",
            "LENGTH",
            "si.metre",
            1,
            1,
            "DEFINED_EXACT",
            (SOURCE_ID,),
        )
    else:
        raise AssertionError("unsupported test length")
    if time == "DAY":
        time_unit = ExactUnitScale(
            "unit.day.86400-si-seconds.defined-exact.v1",
            "TIME",
            "si.second",
            86_400,
            1,
            "DEFINED_EXACT",
            (SOURCE_ID,),
        )
    elif time == "SECOND":
        time_unit = ExactUnitScale(
            "unit.second.defined-exact.v1",
            "TIME",
            "si.second",
            1,
            1,
            "DEFINED_EXACT",
            (SOURCE_ID,),
        )
    else:
        raise AssertionError("unsupported test time")
    mass_unit = ExactUnitScale(
        "unit.kilogram.defined-exact.v1",
        "MASS",
        "si.kilogram",
        1,
        1,
        "DEFINED_EXACT",
        (SOURCE_ID,),
    )
    gm_scale = Fraction(length_unit.numerator, length_unit.denominator) ** 3 / (
        Fraction(time_unit.numerator, time_unit.denominator) ** 2
    )
    gm_unit = ExactUnitScale(
        f"unit.gm.{length.lower()}-per-{time.lower()}.fixture",
        "GRAVITATIONAL_PARAMETER",
        "si.metre3-per-second2",
        gm_scale.numerator,
        gm_scale.denominator,
        "DEFINED_EXACT",
        (SOURCE_ID,),
    )
    return UnitSystemDefinition(
        f"units.{length.lower()}-{time.lower()}.tdb.fixture",
        length_unit,
        time_unit,
        mass_unit,
        gm_unit,
        "TDB",
    )


def spk_query_with_output_units(units: UnitSystemDefinition) -> EphemerisQuerySpec:
    provider, query, _, _ = spk_context()
    replaced_provider = dataclasses.replace(
        provider,
        native_output_unit_system=units,
        content_sha256="",
    )
    return dataclasses.replace(
        query,
        provider=replaced_provider,
        output_unit_system=units,
        content_sha256="",
    )


class ProviderNativeStateSchemaTests(unittest.TestCase):
    def test_exact_exports_field_roster_and_validator_signature(self) -> None:
        self.assertEqual(states_module.__all__, EXPECTED_EXPORTS)
        self.assertEqual(states_module.__all__, sorted(states_module.__all__))
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(ProviderNativeStateBatch)),
            EXPECTED_FIELDS,
        )
        self.assertEqual(
            tuple(inspect.signature(validate_provider_native_state_batch).parameters),
            ("value",),
        )
        from jxplanetx import solar_system

        self.assertEqual(solar_system.__all__, [])

    def test_exact_stage_and_non_authorizing_statuses(self) -> None:
        batch = make_batch()
        self.assertEqual(batch.state_stage, STATE_STAGE)
        self.assertEqual(batch.execution_evidence_status, EXECUTION_STATUS)
        self.assertEqual(batch.target_availability_status, batch.query.target_chain_availability_status)
        self.assertEqual(batch.artifact_custody_status, ARTIFACT_STATUS)
        self.assertEqual(batch.semantic_replay_status, REPLAY_STATUS)
        self.assertEqual(batch.content_integrity_class, INTEGRITY_CLASS)
        self.assertNotIn("EXECUTED", batch.execution_evidence_status)
        self.assertNotIn("VERIFIED", batch.artifact_custody_status)


class ProviderNativeStateSemanticsTests(unittest.TestCase):
    def test_synthetic_fixture_preserves_target_and_component_order(self) -> None:
        batch = make_batch()
        validate_provider_native_state_batch(batch)
        self.assertEqual(batch.target_naif_ids, (-2, -1))
        self.assertEqual(batch.component_order, ("X", "Y", "Z"))
        self.assertEqual(batch.state_shape, (2, 3))
        self.assertEqual(batch.positions[:3], (1.0, 2.0, 3.0))
        self.assertEqual(batch.positions[3:], (4.0, 5.0, 6.0))
        self.assertIsNone(batch.query.observer_naif_id)

    def test_declarative_spk_query_is_not_execution_evidence(self) -> None:
        query = spk_context()[1]
        batch = make_batch(query, batch_id="batch.fixture.spk.declarative")
        validate_provider_native_state_batch(batch)
        self.assertEqual(batch.target_naif_ids, (10, 399))
        self.assertEqual(batch.execution_evidence_status, EXECUTION_STATUS)
        self.assertEqual(batch.artifact_custody_status, ARTIFACT_STATUS)
        self.assertEqual(batch.semantic_replay_status, REPLAY_STATUS)

    def test_et_epoch_seconds_do_not_define_kilometres_per_day_velocity(self) -> None:
        kilometre_day = tdb_output_units(length="KILOMETRE", time="DAY")
        query = spk_query_with_output_units(kilometre_day)
        self.assertEqual(
            query.epoch.coordinate_unit_id,
            "unit.second.defined-exact.v1",
        )
        batch = make_batch(query, batch_id="batch.fixture.km-per-day")
        self.assertEqual(batch.position_unit_id, kilometre_day.length.unit_id)
        self.assertEqual(batch.velocity_length_unit_id, kilometre_day.length.unit_id)
        self.assertEqual(batch.velocity_time_unit_id, kilometre_day.time.unit_id)
        self.assertNotEqual(batch.velocity_time_unit_id, query.epoch.coordinate_unit_id)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                batch,
                velocity_time_unit_id=query.epoch.coordinate_unit_id,
                content_sha256="",
            )

    def test_kilometres_per_second_and_per_day_are_distinct_native_outputs(self) -> None:
        query_day = spk_query_with_output_units(
            tdb_output_units(length="KILOMETRE", time="DAY")
        )
        query_second = spk_query_with_output_units(
            tdb_output_units(length="KILOMETRE", time="SECOND")
        )
        batch_day = make_batch(query_day, batch_id="batch.fixture.km-per-day")
        batch_second = make_batch(query_second, batch_id="batch.fixture.km-per-second")
        self.assertNotEqual(batch_day.velocity_time_unit_id, batch_second.velocity_time_unit_id)
        self.assertNotEqual(batch_day.query.content_sha256, batch_second.query.content_sha256)
        self.assertNotEqual(batch_day.content_sha256, batch_second.content_sha256)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                batch_second,
                velocity_time_unit_id=batch_day.velocity_time_unit_id,
                content_sha256="",
            )

    def test_redundant_query_fields_are_exactly_cross_bound(self) -> None:
        batch = make_batch()
        invalid = (
            {"target_naif_ids": tuple(reversed(batch.target_naif_ids))},
            {"component_order": ("X", "Z", "Y")},
            {"state_shape": (3, 2)},
            {"position_unit_id": "unit.other"},
            {"velocity_length_unit_id": "unit.other"},
            {"velocity_time_unit_id": "unit.other"},
            {"target_availability_status": "VALIDATED"},
        )
        for replacement in invalid:
            with self.subTest(replacement=replacement):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(batch, **replacement, content_sha256="")

    def test_signed_zero_is_preserved_and_bound(self) -> None:
        positive = make_batch(
            positions=(0.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        negative = make_batch(
            positions=(-0.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        self.assertEqual(positive.positions[0].hex(), "0x0.0p+0")
        self.assertEqual(negative.positions[0].hex(), "-0x0.0p+0")
        self.assertNotEqual(positive.content_sha256, negative.content_sha256)

        positive_velocity = make_batch(
            velocities=(0.0, -0.25, -0.375, -0.5, -0.625, -0.75)
        )
        negative_velocity = make_batch(
            velocities=(-0.0, -0.25, -0.375, -0.5, -0.625, -0.75)
        )
        self.assertEqual(positive_velocity.velocities[0].hex(), "0x0.0p+0")
        self.assertEqual(negative_velocity.velocities[0].hex(), "-0x0.0p+0")
        self.assertNotEqual(
            positive_velocity.content_sha256,
            negative_velocity.content_sha256,
        )


class ProviderNativeStateIntegrityTests(unittest.TestCase):
    def test_frozen_m1_serializer_rejects_the_new_contract_directly(self) -> None:
        with self.assertRaises(SolarSystemContractError):
            canonical_json(make_batch())

    def test_literal_production_preimage_and_hash(self) -> None:
        batch = make_batch()
        preimage = (
            BATCH_DOMAIN.encode("ascii")
            + b"\x00"
            + BATCH_SCHEMA.encode("ascii")
            + b"\x00"
            + canonical_json(states_module._batch_payload(batch))
        )
        self.assertEqual(len(preimage), BATCH_LITERAL_LENGTH)
        self.assertEqual(preimage, BATCH_LITERAL_PREIMAGE)
        self.assertEqual(hashlib.sha256(preimage).hexdigest(), BATCH_LITERAL_SHA256)
        self.assertEqual(batch.content_sha256, BATCH_LITERAL_SHA256)

    def test_domain_schema_order_and_signed_zero_mutations_change_digest(self) -> None:
        preimage = BATCH_LITERAL_PREIMAGE
        mutations = (
            preimage.replace(BATCH_DOMAIN.encode("ascii"), b"jx.mutated.domain", 1),
            preimage.replace(BATCH_SCHEMA.encode("ascii"), b"MutatedSchema.v1", 1),
            preimage.replace(b'[["batch_id",', b'[["query",', 1),
            preimage.replace(b'0x1.0000000000000p+0', b'-0x0.0p+0', 1),
        )
        for mutation in mutations:
            with self.subTest(digest=hashlib.sha256(mutation).hexdigest()):
                self.assertNotEqual(hashlib.sha256(mutation).hexdigest(), BATCH_LITERAL_SHA256)

    def test_every_field_and_nested_seal_are_revalidated(self) -> None:
        alternatives: dict[str, object] = {
            "batch_id": "batch.fixture.other",
            "query": dataclasses.replace(
                synthetic_context()[1], query_id="query.fixture.other", content_sha256=""
            ),
            "state_stage": "OTHER_STAGE",
            "execution_evidence_status": "OTHER_EXECUTION_STATUS",
            "target_naif_ids": (-1, -2),
            "component_order": ("Z", "Y", "X"),
            "state_shape": (3, 2),
            "position_unit_id": "unit.other",
            "velocity_length_unit_id": "unit.other",
            "velocity_time_unit_id": "unit.other",
            "velocity_semantics": "OTHER_VELOCITY_SEMANTICS",
            "positions": (9.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            "velocities": (-9.0, -0.25, -0.375, -0.5, -0.625, -0.75),
            "target_availability_status": "OTHER_AVAILABILITY",
            "artifact_custody_status": "OTHER_ARTIFACT_STATUS",
            "semantic_replay_status": "OTHER_REPLAY_STATUS",
            "content_integrity_class": "OTHER_INTEGRITY",
        }
        for field_name, replacement in alternatives.items():
            batch = make_batch()
            object.__setattr__(batch, field_name, replacement)
            with self.subTest(field_name=field_name):
                with self.assertRaises(SolarSystemContractError):
                    validate_provider_native_state_batch(batch)

        stale_query_batch = make_batch()
        object.__setattr__(stale_query_batch.query, "query_id", "query.stale")
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(stale_query_batch)

        stale_epoch_batch = make_batch()
        object.__setattr__(stale_epoch_batch.query.epoch, "whole", 13)
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(stale_epoch_batch)

        stale_provider_batch = make_batch()
        object.__setattr__(
            stale_provider_batch.query.provider.identity,
            "version",
            "stale",
        )
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(stale_provider_batch)

        stale_artifact_batch = make_batch()
        object.__setattr__(
            stale_artifact_batch.query.provider.artifacts[0],
            "artifact_sha256",
            "0" * 64,
        )
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(stale_artifact_batch)

        stale_coverage_batch = make_batch(spk_context()[1])
        stale_coverage = stale_coverage_batch.query.provider.native_frame.coverage
        assert stale_coverage is not None
        object.__setattr__(stale_coverage[0], "endpoint_policy", "CLOSED_OPEN")
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(stale_coverage_batch)

        stale_coverage_epoch_batch = make_batch(spk_context()[1])
        stale_epoch_coverage = (
            stale_coverage_epoch_batch.query.provider.native_frame.coverage
        )
        assert stale_epoch_coverage is not None
        object.__setattr__(stale_epoch_coverage[0].start, "whole", -86_399)
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(stale_coverage_epoch_batch)

        stale_frame_batch = make_batch()
        object.__setattr__(
            stale_frame_batch.query.frame,
            "axes_realization_id",
            "axes.stale",
        )
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(stale_frame_batch)

        stale_units_batch = make_batch()
        object.__setattr__(
            stale_units_batch.query.output_unit_system.time,
            "unit_id",
            "unit.stale",
        )
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(stale_units_batch)

    def test_content_sha_and_subclass_reject(self) -> None:
        batch = make_batch()
        object.__setattr__(batch, "content_sha256", "0" * 64)
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(batch)

        class BatchSubclass(ProviderNativeStateBatch):
            pass

        values = {
            field.name: getattr(make_batch(), field.name)
            for field in dataclasses.fields(ProviderNativeStateBatch)
            if field.name != "content_sha256"
        }
        with self.assertRaises(SolarSystemContractError):
            BatchSubclass(**values)
        with self.assertRaises(SolarSystemContractError):
            validate_provider_native_state_batch(object())  # type: ignore[arg-type]

    def test_supplied_seal_is_preflighted_before_digest_serialization(self) -> None:
        batch = make_batch()
        for invalid_seal in (False, "not-a-sha256"):
            with self.subTest(invalid_seal=invalid_seal):
                with mock.patch.object(
                    states_module,
                    "_batch_digest",
                    side_effect=AssertionError("digest traversal was reached"),
                ) as digest:
                    with self.assertRaises(SolarSystemContractError):
                        dataclasses.replace(
                            batch,
                            content_sha256=invalid_seal,
                        )
                    digest.assert_not_called()


class ProviderNativeStateCapAndTypeTests(unittest.TestCase):
    def test_exact_float_tuple_and_finite_rules(self) -> None:
        batch = make_batch()
        invalid_blocks: tuple[object, ...] = (
            list(batch.positions),
            tuple([1] + list(batch.positions[1:])),
            tuple([True] + list(batch.positions[1:])),
            tuple([float("nan")] + list(batch.positions[1:])),
            tuple([float("inf")] + list(batch.positions[1:])),
        )

        class FloatSubclass(float):
            pass

        invalid_blocks += (
            tuple([FloatSubclass(1.0)] + list(batch.positions[1:])),
        )
        for block in invalid_blocks:
            with self.subTest(block_type=type(block).__name__):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(batch, positions=block, content_sha256="")

    def test_exact_tuple_integer_and_text_types(self) -> None:
        batch = make_batch()
        replacements: tuple[dict[str, object], ...] = (
            {"target_naif_ids": list(batch.target_naif_ids)},
            {"target_naif_ids": (True, -1)},
            {"component_order": ["X", "Y", "Z"]},
            {"state_shape": [2, 3]},
            {"state_shape": (True, 3)},
            {"batch_id": False},
        )
        for replacement in replacements:
            with self.subTest(replacement=replacement):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(batch, **replacement, content_sha256="")

    def test_count_caps_precede_component_traversal(self) -> None:
        class TraversalBomb:
            def __getattribute__(self, name: str) -> object:
                raise AssertionError("component traversed before length rejection")

        batch = make_batch()
        oversized = tuple(TraversalBomb() for _ in range(193))
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(batch, positions=oversized, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(batch, velocities=oversized, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                batch,
                target_naif_ids=(TraversalBomb(), TraversalBomb(), TraversalBomb()),
                content_sha256="",
            )

    def test_maximum_64_target_shape_is_bounded(self) -> None:
        _, query = synthetic_context()
        targets = tuple(range(-64, 0))
        large_query = dataclasses.replace(
            query,
            target_naif_ids=targets,
            content_sha256="",
        )
        values = tuple(float(index) for index in range(192))
        batch = make_batch(
            large_query,
            batch_id="batch.fixture.maximum-targets",
            positions=values,
            velocities=tuple(-value for value in values),
        )
        self.assertEqual(batch.state_shape, (64, 3))
        validate_provider_native_state_batch(batch)

    def test_oversized_batch_id_rejects_before_digest_work(self) -> None:
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                make_batch(),
                batch_id="x" * 257,
                content_sha256="",
            )

    def test_redundant_integer_fields_have_prospective_caps(self) -> None:
        batch = make_batch()
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                batch,
                target_naif_ids=(1 << 4_096, -1),
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                batch,
                state_shape=(1 << 4_096, 3),
                content_sha256="",
            )


class ProviderNativeStateImportBoundaryTests(unittest.TestCase):
    def test_clean_import_is_dependency_and_execution_free(self) -> None:
        code = """
import json, sys
import jxplanetx.solar_system.states as states_module
from jxplanetx import solar_system
forbidden = [
    name for name in (
        'numpy', 'astropy', 'erfa', 'jplephem', 'spiceypy', 'skyfield',
        'rebound', 'socket', 'urllib.request', 'http.client', 'requests'
    ) if name in sys.modules
]
print(json.dumps({
    'exports': states_module.__all__,
    'root_exports': solar_system.__all__,
    'forbidden': forbidden,
}))
"""
        environment = dict(os.environ)
        source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        environment["PYTHONPATH"] = source
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["exports"], EXPECTED_EXPORTS)
        self.assertEqual(payload["root_exports"], [])
        self.assertEqual(payload["forbidden"], [])

    def test_module_nonclaims_are_explicit(self) -> None:
        text = " ".join(states_module.__doc__.lower().split())
        for phrase in (
            "does not execute a provider",
            "open an artifact",
            "segment chain",
            "convert units or time scales",
            "translate an origin",
            "prepare engine arrays",
            "unauthenticated content integrity only",
            "necessary but not sufficient",
            "future provider execution receipt",
            "does not prove target-specific coverage",
            "interpolation or physical-accuracy bounds",
            "enforce network/fallback/extrapolation policy",
            "license, registry, or qualification authority",
        ):
            self.assertIn(phrase, text)


# Locked after the fixture and production schema compile coherently.  These are
# complete independent literal bytes, not output from the production builder.
BATCH_LITERAL_PREIMAGE = base64.b64decode(
    (
        "anhwbGFuZXR4LnNvbGFyLXN5c3RlbS5zdGF0ZXMucHJvdmlkZXItbmF0aXZlLXN0YXRlLWJhdGNoLmNvbnRlbnQtaW50ZWdy"
        "aXR5LnYxAFByb3ZpZGVyTmF0aXZlU3RhdGVCYXRjaC52MQBbImp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uc3RhdGVzLlByb3Zp"
        "ZGVyTmF0aXZlU3RhdGVCYXRjaCIsW1siYmF0Y2hfaWQiLCJiYXRjaC5maXh0dXJlLnN5bnRoZXRpYy52MSJdLFsicXVlcnki"
        "LHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuRXBoZW1lcmlzUXVlcnlTcGVjIiwiZmll"
        "bGRzIjpbWyJxdWVyeV9pZCIsInF1ZXJ5LmZpeHR1cmUuc3ludGhldGljLnYxIl0sWyJwcm92aWRlciIseyJkYXRhY2xhc3Mi"
        "OiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5FcGhlbWVyaXNQcm92aWRlclNwZWMiLCJmaWVsZHMiOltbInNw"
        "ZWNfaWQiLCJwcm92aWRlci5maXh0dXJlLnN5bnRoZXRpYy52MSJdLFsiaWRlbnRpdHkiLHsiZGF0YWNsYXNzIjoianhwbGFu"
        "ZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuUHJvdmlkZXJJZGVudGl0eSIsImZpZWxkcyI6W1sicHJvdmlkZXJfaWQiLCJw"
        "cm92aWRlci5maXh0dXJlIl0sWyJpbXBsZW1lbnRhdGlvbl9pZCIsInByb3ZpZGVyLmZpeHR1cmUuaW1wbGVtZW50YXRpb24u"
        "djEiXSxbInByb3ZpZGVyX2tpbmQiLCJTWU5USEVUSUNfVEVTVCJdLFsidmVyc2lvbiIsIjEiXSxbInJ1bnRpbWVfaWQiLCJj"
        "cHl0aG9uLnRlc3QiXSxbIm1vZHVsZV9uYW1lIiwiZml4dHVyZV9wcm92aWRlciJdLFsiZGlzdHJpYnV0aW9uX25hbWUiLCJm"
        "aXh0dXJlLXByb3ZpZGVyIl0sWyJkaXN0cmlidXRpb25fdmVyc2lvbiIsIjEiXSxbIm9yZGVyZWRfaW1wbGVtZW50YXRpb25f"
        "YXJ0aWZhY3RfaWRzIixbImFydGlmYWN0LnByb3ZpZGVyLnNvdXJjZSJdXSxbImNvbnRlbnRfc2hhMjU2IiwiMGJkOTExNmZj"
        "NGVjYzBjNzhhM2Y1NzlkOWFlODE0N2VmZTlmM2NjOWFlZjE3NjkyN2Q1YzZhN2RlMTE4NWNhMSJdXX1dLFsiYXJ0aWZhY3Rz"
        "IixbeyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5BcnRpZmFjdEJpbmRpbmciLCJmaWVs"
        "ZHMiOltbImFydGlmYWN0X2lkIiwiYXJ0aWZhY3QuZml4dHVyZS5zeW50aGV0aWMiXSxbImFydGlmYWN0X3JvbGUiLCJTWU5U"
        "SEVUSUNfRklYVFVSRSJdLFsicHJvdmlkZXJfaWQiLCJwcm92aWRlci5maXh0dXJlIl0sWyJ2ZXJzaW9uIiwiMSJdLFsibG9n"
        "aWNhbF9sb2NhdG9yIiwicmV0YWluZWQvc3ludGhldGljLWZpeHR1cmUuanNvbiJdLFsibG9jYXRvcl9raW5kIiwiTE9DQUxf"
        "UkVHVUxBUl9GSUxFIl0sWyJieXRlX2xlbmd0aCIsOTZdLFsiYXJ0aWZhY3Rfc2hhMjU2IiwiNDQ0NDQ0NDQ0NDQ0NDQ0NDQ0"
        "NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NCJdLFsibWVkaWFfdHlwZSIsImFwcGxpY2F0"
        "aW9uL2pzb24iXSxbImNvdmVyYWdlX3N0YXR1cyIsIk5PVF9BUFBMSUNBQkxFIl0sWyJjb3ZlcmFnZSIsbnVsbF0sWyJsaWNl"
        "bnNlX2V2aWRlbmNlX3N0YXR1cyIsIk5PVF9BUFBMSUNBQkxFX0lOVEVSTkFMX1NZTlRIRVRJQyJdLFsibGljZW5zZV9zcGR4"
        "IixudWxsXSxbImxpY2Vuc2VfYXJ0aWZhY3RfaWQiLG51bGxdLFsicmVkaXN0cmlidXRpb25fc3RhdHVzIiwiTk9UX0VWQUxV"
        "QVRFRF9CTE9DS0VEIl0sWyJsb2FkX29yZGVyX3N0YXR1cyIsIk5PVF9MT0FEQUJMRSJdLFsibG9hZF9vcmRlciIsbnVsbF0s"
        "WyJleHRyYXBvbGF0aW9uX3BvbGljeSIsIkZPUkJJRCJdLFsiY29udGVudF9pbnRlZ3JpdHlfY2xhc3MiLCJVTkFVVEhFTlRJ"
        "Q0FURURfQ09OVEVOVF9JTlRFR1JJVFlfT05MWSJdLFsiY29udGVudF9zaGEyNTYiLCI4ZGYwNTdhOWM5NjFhNzkxMTYwZTFk"
        "ZDkzM2JmOTA4ZDdhZjU2MmIzMTAyNjhjMWExYTUyNmIxYmFjNTNkNjlkIl1dfSx7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5z"
        "b2xhcl9zeXN0ZW0uY29udHJhY3RzLkFydGlmYWN0QmluZGluZyIsImZpZWxkcyI6W1siYXJ0aWZhY3RfaWQiLCJhcnRpZmFj"
        "dC5wcm92aWRlci5saWNlbnNlIl0sWyJhcnRpZmFjdF9yb2xlIiwiTElDRU5TRSJdLFsicHJvdmlkZXJfaWQiLCJwcm92aWRl"
        "ci5maXh0dXJlIl0sWyJ2ZXJzaW9uIiwiMSJdLFsibG9naWNhbF9sb2NhdG9yIiwicmV0YWluZWQvTElDRU5TRS50eHQiXSxb"
        "ImxvY2F0b3Jfa2luZCIsIkxPQ0FMX1JFR1VMQVJfRklMRSJdLFsiYnl0ZV9sZW5ndGgiLDY0XSxbImFydGlmYWN0X3NoYTI1"
        "NiIsIjMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMiXSxb"
        "Im1lZGlhX3R5cGUiLCJ0ZXh0L3BsYWluIl0sWyJjb3ZlcmFnZV9zdGF0dXMiLCJOT1RfQVBQTElDQUJMRSJdLFsiY292ZXJh"
        "Z2UiLG51bGxdLFsibGljZW5zZV9ldmlkZW5jZV9zdGF0dXMiLCJSRVRBSU5FRF9MSUNFTlNFX1RFWFRfU0VMRl9FVklERU5D"
        "RV9OT05BVVRIT1JJWklORyJdLFsibGljZW5zZV9zcGR4IiwiQlNELTMtQ2xhdXNlIl0sWyJsaWNlbnNlX2FydGlmYWN0X2lk"
        "IixudWxsXSxbInJlZGlzdHJpYnV0aW9uX3N0YXR1cyIsIkJVTkRMRURfV0lUSF9SRVRBSU5FRF9MSUNFTlNFX0VWSURFTkNF"
        "Il0sWyJsb2FkX29yZGVyX3N0YXR1cyIsIk5PVF9MT0FEQUJMRSJdLFsibG9hZF9vcmRlciIsbnVsbF0sWyJleHRyYXBvbGF0"
        "aW9uX3BvbGljeSIsIkZPUkJJRCJdLFsiY29udGVudF9pbnRlZ3JpdHlfY2xhc3MiLCJVTkFVVEhFTlRJQ0FURURfQ09OVEVO"
        "VF9JTlRFR1JJVFlfT05MWSJdLFsiY29udGVudF9zaGEyNTYiLCI0OWM0NWM5NWZlYWYyODI4YjMxNGQ2NDU3OGFmYmNhZjNm"
        "YzQ4NzZlZWZjZWMyMDcxNTkzMzI4NTZhNDhlZmEwIl1dfSx7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0u"
        "Y29udHJhY3RzLkFydGlmYWN0QmluZGluZyIsImZpZWxkcyI6W1siYXJ0aWZhY3RfaWQiLCJhcnRpZmFjdC5wcm92aWRlci5z"
        "b3VyY2UiXSxbImFydGlmYWN0X3JvbGUiLCJTT0ZUV0FSRV9TT1VSQ0UiXSxbInByb3ZpZGVyX2lkIiwicHJvdmlkZXIuZml4"
        "dHVyZSJdLFsidmVyc2lvbiIsIjEiXSxbImxvZ2ljYWxfbG9jYXRvciIsInJldGFpbmVkL2FydGlmYWN0LnByb3ZpZGVyLnNv"
        "dXJjZS5iaW4iXSxbImxvY2F0b3Jfa2luZCIsIkxPQ0FMX1JFR1VMQVJfRklMRSJdLFsiYnl0ZV9sZW5ndGgiLDEyOF0sWyJh"
        "cnRpZmFjdF9zaGEyNTYiLCIxMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTEx"
        "MTExMTExMTExIl0sWyJtZWRpYV90eXBlIiwiYXBwbGljYXRpb24vb2N0ZXQtc3RyZWFtIl0sWyJjb3ZlcmFnZV9zdGF0dXMi"
        "LCJOT1RfQVBQTElDQUJMRSJdLFsiY292ZXJhZ2UiLG51bGxdLFsibGljZW5zZV9ldmlkZW5jZV9zdGF0dXMiLCJSRVRBSU5F"
        "RF9IQVNIX0JPVU5EX0xJQ0VOU0VfQVJUSUZBQ1QiXSxbImxpY2Vuc2Vfc3BkeCIsIkJTRC0zLUNsYXVzZSJdLFsibGljZW5z"
        "ZV9hcnRpZmFjdF9pZCIsImFydGlmYWN0LnByb3ZpZGVyLmxpY2Vuc2UiXSxbInJlZGlzdHJpYnV0aW9uX3N0YXR1cyIsIkJV"
        "TkRMRURfV0lUSF9SRVRBSU5FRF9MSUNFTlNFX0VWSURFTkNFIl0sWyJsb2FkX29yZGVyX3N0YXR1cyIsIk5PVF9MT0FEQUJM"
        "RSJdLFsibG9hZF9vcmRlciIsbnVsbF0sWyJleHRyYXBvbGF0aW9uX3BvbGljeSIsIkZPUkJJRCJdLFsiY29udGVudF9pbnRl"
        "Z3JpdHlfY2xhc3MiLCJVTkFVVEhFTlRJQ0FURURfQ09OVEVOVF9JTlRFR1JJVFlfT05MWSJdLFsiY29udGVudF9zaGEyNTYi"
        "LCIzOWE5OTUxZWFmZjFhZDBjODk3OWJjNGMwMzc0MWI2MDdjZDQyNTkzY2FjYTk0OWVmM2FjNzVlNDI4NDFlMzM3Il1dfV1d"
        "LFsib3JkZXJlZF9sb2FkX2FydGlmYWN0X2lkcyIsW11dLFsiY2FwYWJpbGl0aWVzIixbIkdFT01FVFJJQ19DQVJURVNJQU5f"
        "U1RBVEUiXV0sWyJuYXRpdmVfdGltZV9zY2FsZSIsIlNZTlRIRVRJQyJdLFsibmF0aXZlX2ZyYW1lIix7ImRhdGFjbGFzcyI6"
        "Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkZyYW1lUmVhbGl6YXRpb24iLCJmaWVsZHMiOltbImZyYW1lX2lk"
        "IiwiZnJhbWUuZml4dHVyZS5zeW50aGV0aWMiXSxbImZyYW1lX2tpbmQiLCJTWU5USEVUSUMiXSxbIm9yaWdpbl9raW5kIiwi"
        "U1lOVEhFVElDIl0sWyJvcmlnaW5fbmFpZl9pZCIsbnVsbF0sWyJvcmlnaW5fcmVhbGl6YXRpb25faWQiLCJmaXh0dXJlLnN5"
        "bnRoZXRpYy5vcmlnaW4iXSxbImF4ZXNfcmVhbGl6YXRpb25faWQiLCJmaXh0dXJlLnN5bnRoZXRpYy5heGVzIl0sWyJvcmll"
        "bnRhdGlvbl9tb2RlbF9pZCIsImZpeHR1cmUuc3ludGhldGljLnN0YXRpYyJdLFsib3JpZW50YXRpb25fdGltZV9kZXBlbmRl"
        "bmNlIiwiU1RBVElDIl0sWyJjb29yZGluYXRlX3RpbWVfc2NhbGUiLCJTWU5USEVUSUMiXSxbImNvdmVyYWdlX3N0YXR1cyIs"
        "Ik5PVF9BUFBMSUNBQkxFIl0sWyJjb3ZlcmFnZSIsbnVsbF0sWyJhcnRpZmFjdF9pZHMiLFsiYXJ0aWZhY3QuZml4dHVyZS5z"
        "eW50aGV0aWMiXV0sWyJjb250ZW50X3NoYTI1NiIsIjY1MDAwNDA1ZjE0NDFhZDM1MGU2M2Y1YThkZjQyZmZmYzJkYWE4MGUz"
        "Zjk1ZTZmZjllYmRhMzc3NGU0MzFkOWIiXV19XSxbIm5hdGl2ZV9vdXRwdXRfdW5pdF9zeXN0ZW0iLHsiZGF0YWNsYXNzIjoi"
        "anhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuVW5pdFN5c3RlbURlZmluaXRpb24iLCJmaWVsZHMiOltbInVuaXRf"
        "c3lzdGVtX2lkIiwidW5pdHMuc3ludGhldGljIl0sWyJsZW5ndGgiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5"
        "c3RlbS5jb250cmFjdHMuRXhhY3RVbml0U2NhbGUiLCJmaWVsZHMiOltbInVuaXRfaWQiLCJ1bml0Lm1ldHJlIl0sWyJkaW1l"
        "bnNpb24iLCJMRU5HVEgiXSxbInNpX3VuaXRfaWQiLCJzaS5tZXRyZSJdLFsibnVtZXJhdG9yIiwxXSxbImRlbm9taW5hdG9y"
        "IiwxXSxbImRlZmluaXRpb25fY2xhc3NpZmljYXRpb24iLCJERUZJTkVEX0VYQUNUIl0sWyJhcnRpZmFjdF9pZHMiLFsiYXJ0"
        "aWZhY3QucHJvdmlkZXIuc291cmNlIl1dLFsiY29udGVudF9zaGEyNTYiLCJkNWNmMzRiNjc5OTY3Yjk2MjFmOTY2NjViYmRl"
        "YzQ5ZDJiOTg1YjViMDJlMDk5YjMyMWY1MDZhODY2ZDZlNmZjIl1dfV0sWyJ0aW1lIix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0"
        "eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkV4YWN0VW5pdFNjYWxlIiwiZmllbGRzIjpbWyJ1bml0X2lkIiwidW5pdC5zeW50"
        "aGV0aWMuZHVyYXRpb24iXSxbImRpbWVuc2lvbiIsIlRJTUUiXSxbInNpX3VuaXRfaWQiLCJzaS5zZWNvbmQiXSxbIm51bWVy"
        "YXRvciIsMV0sWyJkZW5vbWluYXRvciIsMV0sWyJkZWZpbml0aW9uX2NsYXNzaWZpY2F0aW9uIiwiREVGSU5FRF9FWEFDVCJd"
        "LFsiYXJ0aWZhY3RfaWRzIixbImFydGlmYWN0LnByb3ZpZGVyLnNvdXJjZSJdXSxbImNvbnRlbnRfc2hhMjU2IiwiMjIyY2Fi"
        "YWI0ZGY4ODIwNjg4MDg5MWM3MmE0MjFhN2QzMmVmN2ZkNmQyYTk3NWNiYzMxZTU4ZTEyNzY0YWMxYiJdXX1dLFsibWFzcyIs"
        "eyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5FeGFjdFVuaXRTY2FsZSIsImZpZWxkcyI6"
        "W1sidW5pdF9pZCIsInVuaXQua2lsb2dyYW0iXSxbImRpbWVuc2lvbiIsIk1BU1MiXSxbInNpX3VuaXRfaWQiLCJzaS5raWxv"
        "Z3JhbSJdLFsibnVtZXJhdG9yIiwxXSxbImRlbm9taW5hdG9yIiwxXSxbImRlZmluaXRpb25fY2xhc3NpZmljYXRpb24iLCJE"
        "RUZJTkVEX0VYQUNUIl0sWyJhcnRpZmFjdF9pZHMiLFsiYXJ0aWZhY3QucHJvdmlkZXIuc291cmNlIl1dLFsiY29udGVudF9z"
        "aGEyNTYiLCI0OTY1ZGJiMjg3ZmE1ZjcxYmNlZTA4ODkxZTQzNzBlMmYzYzU4Y2UzMGJiZWE1YjQyZmRiNjE1NDkyZTRhYjA3"
        "Il1dfV0sWyJncmF2aXRhdGlvbmFsX3BhcmFtZXRlciIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNv"
        "bnRyYWN0cy5FeGFjdFVuaXRTY2FsZSIsImZpZWxkcyI6W1sidW5pdF9pZCIsInVuaXQubWV0cmUzLXBlci1zZWNvbmQyIl0s"
        "WyJkaW1lbnNpb24iLCJHUkFWSVRBVElPTkFMX1BBUkFNRVRFUiJdLFsic2lfdW5pdF9pZCIsInNpLm1ldHJlMy1wZXItc2Vj"
        "b25kMiJdLFsibnVtZXJhdG9yIiwxXSxbImRlbm9taW5hdG9yIiwxXSxbImRlZmluaXRpb25fY2xhc3NpZmljYXRpb24iLCJE"
        "RUZJTkVEX0VYQUNUIl0sWyJhcnRpZmFjdF9pZHMiLFsiYXJ0aWZhY3QucHJvdmlkZXIuc291cmNlIl1dLFsiY29udGVudF9z"
        "aGEyNTYiLCIwODFlZmZjMzFlZTU1M2MxMmM1NmI4NWM2Zjc5MmRkODAxNDQ1OWIwZTkxOGY1MTgyNmFiMzgxNTBhYTFjZmE3"
        "Il1dfV0sWyJjb29yZGluYXRlX3RpbWVfc2NhbGUiLCJTWU5USEVUSUMiXSxbImNvbnRlbnRfc2hhMjU2IiwiZTZjOTg2Mzdh"
        "N2MwMjllNGNmZGE5ODI5NTc3ZGY0MzE1YzgwM2YyMDdkYzNlMGQ5YjY3MDE0MGE4YzA3YWU2NSJdXX1dLFsibmV0d29ya19h"
        "Y2Nlc3MiLGZhbHNlXSxbImZhbGxiYWNrX2FsbG93ZWQiLGZhbHNlXSxbImV4dHJhcG9sYXRpb25fcG9saWN5IiwiRk9SQklE"
        "Il0sWyJldmlkZW5jZV9jbGFzcyIsIk1PREVMX09VVFBVVCJdLFsicmVnaXN0cnlfYXV0aG9yaXplZCIsZmFsc2VdLFsicXVh"
        "bGlmaWNhdGlvbl9hdXRob3JpemVkIixmYWxzZV0sWyJjb250ZW50X3NoYTI1NiIsIjQ4YzNhMDVlZDM3NzBmOTIwYzBhNDUx"
        "M2YwNWY1NGJjMTAwYmRiOWZlZmU2NDVlNzFlNzJiNGU5OWFmMTkzZmEiXV19XSxbImVwb2NoIix7ImRhdGFjbGFzcyI6Imp4"
        "cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkNvb3JkaW5hdGVFcG9jaCIsImZpZWxkcyI6W1sidGltZV9zY2FsZSIs"
        "IlNZTlRIRVRJQyJdLFsicmVwcmVzZW50YXRpb24iLCJTWU5USEVUSUNfT0ZGU0VUIl0sWyJ3aG9sZSIsMTJdLFsiZnJhY3Rp"
        "b24iLHsiZmxvYXRfaGV4IjoiMHgxLjAwMDAwMDAwMDAwMDBwLTIifV0sWyJjb29yZGluYXRlX3VuaXRfaWQiLCJ1bml0LnN5"
        "bnRoZXRpYy5kdXJhdGlvbiJdLFsib3JpZ2luX2lkIiwicHJvdmlkZXIuZml4dHVyZS5zeW50aGV0aWMudjEuZXBvY2gtemVy"
        "byJdLFsicmVhbGl6YXRpb25faWQiLCJwcm92aWRlci5maXh0dXJlLnN5bnRoZXRpYy52MS5lcG9jaC1yZWFsaXphdGlvbiJd"
        "LFsiYXJ0aWZhY3RfaWRzIixbXV0sWyJjb250ZW50X3NoYTI1NiIsIjNkZDNjNDg3MDI5MDMxNDZiNTA0ODllOGQ1OTY2ODZi"
        "MWNiOTUzNjcwNjk5NjgyYmI0ZjMxNjM2N2ExZDQyMzAiXV19XSxbInRhcmdldF9uYWlmX2lkcyIsWy0yLC0xXV0sWyJvYnNl"
        "cnZlcl9uYWlmX2lkIixudWxsXSxbImZyYW1lIix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJh"
        "Y3RzLkZyYW1lUmVhbGl6YXRpb24iLCJmaWVsZHMiOltbImZyYW1lX2lkIiwiZnJhbWUuZml4dHVyZS5zeW50aGV0aWMiXSxb"
        "ImZyYW1lX2tpbmQiLCJTWU5USEVUSUMiXSxbIm9yaWdpbl9raW5kIiwiU1lOVEhFVElDIl0sWyJvcmlnaW5fbmFpZl9pZCIs"
        "bnVsbF0sWyJvcmlnaW5fcmVhbGl6YXRpb25faWQiLCJmaXh0dXJlLnN5bnRoZXRpYy5vcmlnaW4iXSxbImF4ZXNfcmVhbGl6"
        "YXRpb25faWQiLCJmaXh0dXJlLnN5bnRoZXRpYy5heGVzIl0sWyJvcmllbnRhdGlvbl9tb2RlbF9pZCIsImZpeHR1cmUuc3lu"
        "dGhldGljLnN0YXRpYyJdLFsib3JpZW50YXRpb25fdGltZV9kZXBlbmRlbmNlIiwiU1RBVElDIl0sWyJjb29yZGluYXRlX3Rp"
        "bWVfc2NhbGUiLCJTWU5USEVUSUMiXSxbImNvdmVyYWdlX3N0YXR1cyIsIk5PVF9BUFBMSUNBQkxFIl0sWyJjb3ZlcmFnZSIs"
        "bnVsbF0sWyJhcnRpZmFjdF9pZHMiLFsiYXJ0aWZhY3QuZml4dHVyZS5zeW50aGV0aWMiXV0sWyJjb250ZW50X3NoYTI1NiIs"
        "IjY1MDAwNDA1ZjE0NDFhZDM1MGU2M2Y1YThkZjQyZmZmYzJkYWE4MGUzZjk1ZTZmZjllYmRhMzc3NGU0MzFkOWIiXV19XSxb"
        "ImFiZXJyYXRpb25fY29ycmVjdGlvbiIsIk5PTkUiXSxbIm91dHB1dF91bml0X3N5c3RlbSIseyJkYXRhY2xhc3MiOiJqeHBs"
        "YW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5Vbml0U3lzdGVtRGVmaW5pdGlvbiIsImZpZWxkcyI6W1sidW5pdF9zeXN0"
        "ZW1faWQiLCJ1bml0cy5zeW50aGV0aWMiXSxbImxlbmd0aCIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVt"
        "LmNvbnRyYWN0cy5FeGFjdFVuaXRTY2FsZSIsImZpZWxkcyI6W1sidW5pdF9pZCIsInVuaXQubWV0cmUiXSxbImRpbWVuc2lv"
        "biIsIkxFTkdUSCJdLFsic2lfdW5pdF9pZCIsInNpLm1ldHJlIl0sWyJudW1lcmF0b3IiLDFdLFsiZGVub21pbmF0b3IiLDFd"
        "LFsiZGVmaW5pdGlvbl9jbGFzc2lmaWNhdGlvbiIsIkRFRklORURfRVhBQ1QiXSxbImFydGlmYWN0X2lkcyIsWyJhcnRpZmFj"
        "dC5wcm92aWRlci5zb3VyY2UiXV0sWyJjb250ZW50X3NoYTI1NiIsImQ1Y2YzNGI2Nzk5NjdiOTYyMWY5NjY2NWJiZGVjNDlk"
        "MmI5ODViNWIwMmUwOTliMzIxZjUwNmE4NjZkNmU2ZmMiXV19XSxbInRpbWUiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNv"
        "bGFyX3N5c3RlbS5jb250cmFjdHMuRXhhY3RVbml0U2NhbGUiLCJmaWVsZHMiOltbInVuaXRfaWQiLCJ1bml0LnN5bnRoZXRp"
        "Yy5kdXJhdGlvbiJdLFsiZGltZW5zaW9uIiwiVElNRSJdLFsic2lfdW5pdF9pZCIsInNpLnNlY29uZCJdLFsibnVtZXJhdG9y"
        "IiwxXSxbImRlbm9taW5hdG9yIiwxXSxbImRlZmluaXRpb25fY2xhc3NpZmljYXRpb24iLCJERUZJTkVEX0VYQUNUIl0sWyJh"
        "cnRpZmFjdF9pZHMiLFsiYXJ0aWZhY3QucHJvdmlkZXIuc291cmNlIl1dLFsiY29udGVudF9zaGEyNTYiLCIyMjJjYWJhYjRk"
        "Zjg4MjA2ODgwODkxYzcyYTQyMWE3ZDMyZWY3ZmQ2ZDJhOTc1Y2JjMzFlNThlMTI3NjRhYzFiIl1dfV0sWyJtYXNzIix7ImRh"
        "dGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkV4YWN0VW5pdFNjYWxlIiwiZmllbGRzIjpbWyJ1"
        "bml0X2lkIiwidW5pdC5raWxvZ3JhbSJdLFsiZGltZW5zaW9uIiwiTUFTUyJdLFsic2lfdW5pdF9pZCIsInNpLmtpbG9ncmFt"
        "Il0sWyJudW1lcmF0b3IiLDFdLFsiZGVub21pbmF0b3IiLDFdLFsiZGVmaW5pdGlvbl9jbGFzc2lmaWNhdGlvbiIsIkRFRklO"
        "RURfRVhBQ1QiXSxbImFydGlmYWN0X2lkcyIsWyJhcnRpZmFjdC5wcm92aWRlci5zb3VyY2UiXV0sWyJjb250ZW50X3NoYTI1"
        "NiIsIjQ5NjVkYmIyODdmYTVmNzFiY2VlMDg4OTFlNDM3MGUyZjNjNThjZTMwYmJlYTViNDJmZGI2MTU0OTJlNGFiMDciXV19"
        "XSxbImdyYXZpdGF0aW9uYWxfcGFyYW1ldGVyIix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJh"
        "Y3RzLkV4YWN0VW5pdFNjYWxlIiwiZmllbGRzIjpbWyJ1bml0X2lkIiwidW5pdC5tZXRyZTMtcGVyLXNlY29uZDIiXSxbImRp"
        "bWVuc2lvbiIsIkdSQVZJVEFUSU9OQUxfUEFSQU1FVEVSIl0sWyJzaV91bml0X2lkIiwic2kubWV0cmUzLXBlci1zZWNvbmQy"
        "Il0sWyJudW1lcmF0b3IiLDFdLFsiZGVub21pbmF0b3IiLDFdLFsiZGVmaW5pdGlvbl9jbGFzc2lmaWNhdGlvbiIsIkRFRklO"
        "RURfRVhBQ1QiXSxbImFydGlmYWN0X2lkcyIsWyJhcnRpZmFjdC5wcm92aWRlci5zb3VyY2UiXV0sWyJjb250ZW50X3NoYTI1"
        "NiIsIjA4MWVmZmMzMWVlNTUzYzEyYzU2Yjg1YzZmNzkyZGQ4MDE0NDU5YjBlOTE4ZjUxODI2YWIzODE1MGFhMWNmYTciXV19"
        "XSxbImNvb3JkaW5hdGVfdGltZV9zY2FsZSIsIlNZTlRIRVRJQyJdLFsiY29udGVudF9zaGEyNTYiLCJlNmM5ODYzN2E3YzAy"
        "OWU0Y2ZkYTk4Mjk1NzdkZjQzMTVjODAzZjIwN2RjM2UwZDliNjcwMTQwYThjMDdhZTY1Il1dfV0sWyJzdGF0ZV9raW5kIiwi"
        "R0VPTUVUUklDIl0sWyJ0YXJnZXRfY2hhaW5fYXZhaWxhYmlsaXR5X3N0YXR1cyIsIlJFUVVJUkVTX1JVTlRJTUVfUFJPVklE"
        "RVJfVEFSR0VUX0FWQUlMQUJJTElUWV9WQUxJREFUSU9OIl0sWyJjb250ZW50X3NoYTI1NiIsIjI2NWU4NDVkOTFhYzE2ODhl"
        "MmNjMjg5Yzg0MmJjZjBlMGUwNDg2MWU2YmUxOGM1ZjFjMDVjOGVmMjllYTQ1M2YiXV19XSxbInN0YXRlX3N0YWdlIiwiREVD"
        "TEFSRURfUFJPVklERVJfTkFUSVZFX0dFT01FVFJJQ19UQVJHRVRfUkVMQVRJVkVfT0JTRVJWRVJfU1RBVEVfTk9fSlhfQ09O"
        "VkVSU0lPTiJdLFsiZXhlY3V0aW9uX2V2aWRlbmNlX3N0YXR1cyIsIkNBTExFUl9TVVBQTElFRF9VTkFVVEhFTlRJQ0FURURf"
        "Tk9fRVhFQ1VUSU9OX1JFQ0VJUFQiXSxbInRhcmdldF9uYWlmX2lkcyIsWy0yLC0xXV0sWyJjb21wb25lbnRfb3JkZXIiLFsi"
        "WCIsIlkiLCJaIl1dLFsic3RhdGVfc2hhcGUiLFsyLDNdXSxbInBvc2l0aW9uX3VuaXRfaWQiLCJ1bml0Lm1ldHJlIl0sWyJ2"
        "ZWxvY2l0eV9sZW5ndGhfdW5pdF9pZCIsInVuaXQubWV0cmUiXSxbInZlbG9jaXR5X3RpbWVfdW5pdF9pZCIsInVuaXQuc3lu"
        "dGhldGljLmR1cmF0aW9uIl0sWyJ2ZWxvY2l0eV9zZW1hbnRpY3MiLCJDT09SRElOQVRFX0RFUklWQVRJVkVfUEVSX1BST1ZJ"
        "REVSX05BVElWRV9USU1FX1VOSVQiXSxbInBvc2l0aW9ucyIsW3siZmxvYXRfaGV4IjoiMHgxLjAwMDAwMDAwMDAwMDBwKzAi"
        "fSx7ImZsb2F0X2hleCI6IjB4MS4wMDAwMDAwMDAwMDAwcCsxIn0seyJmbG9hdF9oZXgiOiIweDEuODAwMDAwMDAwMDAwMHAr"
        "MSJ9LHsiZmxvYXRfaGV4IjoiMHgxLjAwMDAwMDAwMDAwMDBwKzIifSx7ImZsb2F0X2hleCI6IjB4MS40MDAwMDAwMDAwMDAw"
        "cCsyIn0seyJmbG9hdF9oZXgiOiIweDEuODAwMDAwMDAwMDAwMHArMiJ9XV0sWyJ2ZWxvY2l0aWVzIixbeyJmbG9hdF9oZXgi"
        "OiItMHgxLjAwMDAwMDAwMDAwMDBwLTMifSx7ImZsb2F0X2hleCI6Ii0weDEuMDAwMDAwMDAwMDAwMHAtMiJ9LHsiZmxvYXRf"
        "aGV4IjoiLTB4MS44MDAwMDAwMDAwMDAwcC0yIn0seyJmbG9hdF9oZXgiOiItMHgxLjAwMDAwMDAwMDAwMDBwLTEifSx7ImZs"
        "b2F0X2hleCI6Ii0weDEuNDAwMDAwMDAwMDAwMHAtMSJ9LHsiZmxvYXRfaGV4IjoiLTB4MS44MDAwMDAwMDAwMDAwcC0xIn1d"
        "XSxbInRhcmdldF9hdmFpbGFiaWxpdHlfc3RhdHVzIiwiUkVRVUlSRVNfUlVOVElNRV9QUk9WSURFUl9UQVJHRVRfQVZBSUxB"
        "QklMSVRZX1ZBTElEQVRJT04iXSxbImFydGlmYWN0X2N1c3RvZHlfc3RhdHVzIiwiTk9fQVJUSUZBQ1RfQVRfVVNFX0NVU1RP"
        "RFlfRVZJREVOQ0UiXSxbInNlbWFudGljX3JlcGxheV9zdGF0dXMiLCJSRVFVSVJFU19QUk9WSURFUl9TUEVDSUZJQ19TRU1B"
        "TlRJQ19SRVBMQVkiXSxbImNvbnRlbnRfaW50ZWdyaXR5X2NsYXNzIiwiVU5BVVRIRU5USUNBVEVEX0NPTlRFTlRfSU5URUdS"
        "SVRZX09OTFkiXV1d"
    ).encode("ascii"),
    validate=True,
)
BATCH_LITERAL_LENGTH = 11_172
BATCH_LITERAL_SHA256 = (
    "60b0e59e6efa76cea7999894c13df4b26162ae18f602e174c58e914e344ce1b2"
)


if __name__ == "__main__":
    unittest.main()
