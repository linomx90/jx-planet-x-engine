"""Exact schema, integrity, and cap tests for Solar-System Milestone 1."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from jxplanetx.solar_system import contracts as contracts_module
from jxplanetx.solar_system import serialization as serialization_module
from jxplanetx.solar_system.contracts import (
    ArtifactBinding,
    BodyParameter,
    CoordinateEpoch,
    CoverageInterval,
    EphemerisProviderSpec,
    EphemerisQuerySpec,
    ExactUnitScale,
    FrameRealization,
    PhysicalConstant,
    ProviderIdentity,
    SolarSystemContractError,
    SolarSystemCoverageError,
    UnitSystemDefinition,
    validate_integrity,
)
from jxplanetx.solar_system.serialization import canonical_json, domain_sha256


DAY_UNIT = "unit.day.86400-si-seconds.defined-exact.v1"
SECOND_UNIT = "unit.second.defined-exact.v1"
SOURCE_ID = "artifact.provider.source"
SPK_ID = "artifact.fixture.spk"
LICENSE_ID = "artifact.provider.license"


def external_artifact(
    *,
    artifact_id: str,
    role: str,
    version: str = "1",
    load_order: int | None = None,
    coverage: tuple[CoverageInterval, ...] | None = None,
) -> ArtifactBinding:
    is_implementation = role in (
        "NATIVE_LIBRARY",
        "SOFTWARE_DISTRIBUTION",
        "SOFTWARE_SOURCE",
    )
    return ArtifactBinding(
        artifact_id=artifact_id,
        artifact_role=role,
        provider_id="provider.fixture",
        version=version,
        logical_locator=f"retained/{artifact_id}.bin",
        locator_kind="LOCAL_REGULAR_FILE",
        byte_length=128,
        artifact_sha256=("1" if artifact_id == SOURCE_ID else "2") * 64,
        media_type="application/octet-stream",
        coverage_status=(
            "COARSE_ARTIFACT_TIME_ENVELOPE" if coverage is not None else "NOT_APPLICABLE"
        ),
        coverage=coverage,
        license_evidence_status=(
            "RETAINED_HASH_BOUND_LICENSE_ARTIFACT"
            if is_implementation
            else "EXTERNAL_NOT_REDISTRIBUTED_LICENSE_NOT_RETAINED_NONAUTHORIZING"
        ),
        license_spdx="BSD-3-Clause",
        license_artifact_id=LICENSE_ID if is_implementation else None,
        redistribution_status=(
            "BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE"
            if is_implementation
            else "EXTERNAL_REFERENCE_NOT_REDISTRIBUTED"
        ),
        load_order_status=(
            "ORDERED_LOAD_MEMBER" if load_order is not None else "NOT_LOADABLE"
        ),
        load_order=load_order,
        extrapolation_policy="FORBID",
        content_integrity_class="UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY",
    )


def license_artifact() -> ArtifactBinding:
    return ArtifactBinding(
        artifact_id=LICENSE_ID,
        artifact_role="LICENSE",
        provider_id="provider.fixture",
        version="1",
        logical_locator="retained/LICENSE.txt",
        locator_kind="LOCAL_REGULAR_FILE",
        byte_length=64,
        artifact_sha256="3" * 64,
        media_type="text/plain",
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        license_evidence_status="RETAINED_LICENSE_TEXT_SELF_EVIDENCE_NONAUTHORIZING",
        license_spdx="BSD-3-Clause",
        license_artifact_id=None,
        redistribution_status="BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE",
        load_order_status="NOT_LOADABLE",
        load_order=None,
        extrapolation_policy="FORBID",
        content_integrity_class="UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY",
    )


def synthetic_fixture_artifact() -> ArtifactBinding:
    return ArtifactBinding(
        artifact_id="artifact.fixture.synthetic",
        artifact_role="SYNTHETIC_FIXTURE",
        provider_id="provider.fixture",
        version="1",
        logical_locator="retained/synthetic-fixture.json",
        locator_kind="LOCAL_REGULAR_FILE",
        byte_length=96,
        artifact_sha256="4" * 64,
        media_type="application/json",
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        license_evidence_status="NOT_APPLICABLE_INTERNAL_SYNTHETIC",
        license_spdx=None,
        license_artifact_id=None,
        redistribution_status="NOT_EVALUATED_BLOCKED",
        load_order_status="NOT_LOADABLE",
        load_order=None,
        extrapolation_policy="FORBID",
        content_integrity_class="UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY",
    )


def exact_units(*, scale: str, artifact_ids: tuple[str, ...]) -> UnitSystemDefinition:
    length = ExactUnitScale(
        "unit.metre",
        "LENGTH",
        "si.metre",
        1,
        1,
        "DEFINED_EXACT",
        artifact_ids,
    )
    time = ExactUnitScale(
        SECOND_UNIT if scale == "TDB" else "unit.synthetic.duration",
        "TIME",
        "si.second",
        1,
        1,
        "DEFINED_EXACT",
        artifact_ids,
    )
    mass = ExactUnitScale(
        "unit.kilogram",
        "MASS",
        "si.kilogram",
        1,
        1,
        "DEFINED_EXACT",
        artifact_ids,
    )
    gm = ExactUnitScale(
        "unit.metre3-per-second2",
        "GRAVITATIONAL_PARAMETER",
        "si.metre3-per-second2",
        1,
        1,
        "DEFINED_EXACT",
        artifact_ids,
    )
    return UnitSystemDefinition(
        "units.si-tdb" if scale == "TDB" else "units.synthetic",
        length,
        time,
        mass,
        gm,
        scale,
    )


def provider_identity(*, kind: str = "LOCAL_OFFLINE_SPK") -> ProviderIdentity:
    return ProviderIdentity(
        provider_id="provider.fixture",
        implementation_id="provider.fixture.implementation.v1",
        provider_kind=kind,
        version="1",
        runtime_id="cpython.test",
        module_name="fixture_provider",
        distribution_name="fixture-provider",
        distribution_version="1",
        ordered_implementation_artifact_ids=(SOURCE_ID,),
    )


def physical_epoch(
    whole: int,
    fraction: float,
    *,
    artifact_ids: tuple[str, ...] = (SOURCE_ID,),
) -> CoordinateEpoch:
    return CoordinateEpoch(
        "TDB",
        "JD_TWO_PART",
        whole,
        fraction,
        DAY_UNIT,
        "JULIAN_DATE",
        "fixture.tdb-realization",
        artifact_ids,
    )


def spk_epoch(
    whole: int,
    fraction: float,
    *,
    artifact_ids: tuple[str, ...] = (),
) -> CoordinateEpoch:
    return CoordinateEpoch(
        "TDB",
        "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART",
        whole,
        fraction,
        SECOND_UNIT,
        "SPICE_J2000_TDB_ORIGIN",
        "fixture.spk-et-realization",
        artifact_ids,
    )


def spk_context() -> tuple[
    EphemerisProviderSpec,
    EphemerisQuerySpec,
    CoverageInterval,
    UnitSystemDefinition,
]:
    start = spk_epoch(-86_400, 0.0)
    end = spk_epoch(86_400, 0.0)
    coverage = CoverageInterval(start, end, "CLOSED_CLOSED")
    source = external_artifact(artifact_id=SOURCE_ID, role="SOFTWARE_SOURCE")
    license_text = license_artifact()
    spk = external_artifact(
        artifact_id=SPK_ID,
        role="SPK",
        load_order=0,
        coverage=(coverage,),
    )
    frame = FrameRealization(
        frame_id="frame.fixture.de-icrf",
        frame_kind="TDB_COMPATIBLE_BARYCENTRIC_INERTIAL",
        origin_kind="SOLAR_SYSTEM_BARYCENTER",
        origin_naif_id=0,
        origin_realization_id="fixture.de-release.ssb",
        axes_realization_id="fixture.de-release.icrf",
        orientation_model_id="fixture.de-release.orientation",
        orientation_time_dependence="STATIC",
        coordinate_time_scale="TDB",
        coverage_status="RETAINED",
        coverage=(coverage,),
        artifact_ids=(SPK_ID,),
    )
    units = exact_units(scale="TDB", artifact_ids=(SOURCE_ID,))
    provider = EphemerisProviderSpec(
        spec_id="provider.fixture.spec.v1",
        identity=provider_identity(),
        artifacts=(spk, license_text, source),
        ordered_load_artifact_ids=(SPK_ID,),
        capabilities=(
            "COARSE_ARTIFACT_TIME_ENVELOPE",
            "GEOMETRIC_CARTESIAN_STATE",
            "TDB_COORDINATE_EPOCH",
        ),
        native_time_scale="TDB",
        native_frame=frame,
        native_output_unit_system=units,
        network_access=False,
        fallback_allowed=False,
        extrapolation_policy="FORBID",
        evidence_class="MODEL_OUTPUT",
        registry_authorized=False,
        qualification_authorized=False,
    )
    query = EphemerisQuerySpec(
        query_id="query.fixture.v1",
        provider=provider,
        epoch=spk_epoch(0, 0.25),
        target_naif_ids=(10, 399),
        observer_naif_id=0,
        frame=frame,
        aberration_correction="NONE",
        output_unit_system=units,
        state_kind="GEOMETRIC",
        target_chain_availability_status=(
            "REQUIRES_RUNTIME_PROVIDER_TARGET_AVAILABILITY_VALIDATION"
        ),
    )
    return provider, query, coverage, units


def synthetic_context() -> tuple[EphemerisProviderSpec, EphemerisQuerySpec]:
    source = external_artifact(artifact_id=SOURCE_ID, role="SOFTWARE_SOURCE")
    license_text = license_artifact()
    fixture = synthetic_fixture_artifact()
    units = exact_units(scale="SYNTHETIC", artifact_ids=(SOURCE_ID,))
    frame = FrameRealization(
        frame_id="frame.fixture.synthetic",
        frame_kind="SYNTHETIC",
        origin_kind="SYNTHETIC",
        origin_naif_id=None,
        origin_realization_id="fixture.synthetic.origin",
        axes_realization_id="fixture.synthetic.axes",
        orientation_model_id="fixture.synthetic.static",
        orientation_time_dependence="STATIC",
        coordinate_time_scale="SYNTHETIC",
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        artifact_ids=("artifact.fixture.synthetic",),
    )
    provider = EphemerisProviderSpec(
        spec_id="provider.fixture.synthetic.v1",
        identity=provider_identity(kind="SYNTHETIC_TEST"),
        artifacts=(fixture, license_text, source),
        ordered_load_artifact_ids=(),
        capabilities=("GEOMETRIC_CARTESIAN_STATE",),
        native_time_scale="SYNTHETIC",
        native_frame=frame,
        native_output_unit_system=units,
        network_access=False,
        fallback_allowed=False,
        extrapolation_policy="FORBID",
        evidence_class="MODEL_OUTPUT",
        registry_authorized=False,
        qualification_authorized=False,
    )
    epoch = CoordinateEpoch(
        "SYNTHETIC",
        "SYNTHETIC_OFFSET",
        12,
        0.25,
        units.time.unit_id,
        "provider.fixture.synthetic.v1.epoch-zero",
        "provider.fixture.synthetic.v1.epoch-realization",
        (),
    )
    query = EphemerisQuerySpec(
        query_id="query.fixture.synthetic.v1",
        provider=provider,
        epoch=epoch,
        target_naif_ids=(-2, -1),
        observer_naif_id=None,
        frame=frame,
        aberration_correction="NONE",
        output_unit_system=units,
        state_kind="GEOMETRIC",
        target_chain_availability_status=(
            "REQUIRES_RUNTIME_PROVIDER_TARGET_AVAILABILITY_VALIDATION"
        ),
    )
    return provider, query


def constants_and_body() -> tuple[PhysicalConstant, PhysicalConstant, PhysicalConstant, BodyParameter]:
    units = exact_units(scale="TDB", artifact_ids=(SOURCE_ID,))
    gm = PhysicalConstant(
        constant_id="gm.star",
        quantity_kind="GRAVITATIONAL_PARAMETER",
        source_decimal="132712440018000000000",
        operational_value=float("1.32712440018e20"),
        unit=units.gravitational_parameter,
        classification="FITTED",
        uncertainty_status="NOT_PROVIDED",
        uncertainty=None,
        uncertainty_kind=None,
        uncertainty_scale="NOT_APPLICABLE",
        confidence_level=None,
        covariance_group=None,
        coordinate_scale_applicability="TDB_COMPATIBLE",
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        artifact_ids=(SOURCE_ID,),
    )
    mass = PhysicalConstant(
        constant_id="mass.star",
        quantity_kind="MASS",
        source_decimal="1988409870698051000000000000000",
        operational_value=float("1.988409870698051e30"),
        unit=units.mass,
        classification="ESTIMATED",
        uncertainty_status="PROVIDED",
        uncertainty=1.0e20,
        uncertainty_kind="STANDARD_UNCERTAINTY",
        uncertainty_scale="ABSOLUTE_SAME_UNIT_AS_VALUE",
        confidence_level=None,
        covariance_group="fixture.star.mass",
        coordinate_scale_applicability="NOT_APPLICABLE",
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        artifact_ids=(SOURCE_ID,),
    )
    radius = PhysicalConstant(
        constant_id="radius.star",
        quantity_kind="RADIUS",
        source_decimal="695700000",
        operational_value=695_700_000.0,
        unit=units.length,
        classification="NOMINAL_EXACT",
        uncertainty_status="EXACT_NOT_APPLICABLE",
        uncertainty=None,
        uncertainty_kind=None,
        uncertainty_scale="NOT_APPLICABLE",
        confidence_level=None,
        covariance_group=None,
        coordinate_scale_applicability="NOT_APPLICABLE",
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        artifact_ids=(SOURCE_ID,),
    )
    body = BodyParameter(
        body_id="SUN",
        naif_id=10,
        body_role="MASSIVE_BODY_CENTER",
        gravitational_parameter_status="PROVIDED_DYNAMICS_PARAMETER",
        gravitational_parameter=gm,
        mass_status="PROVIDED_NOT_USED_BY_GM_DYNAMICS",
        mass=mass,
        radius_status="PROVIDED_REFERENCE_RADIUS",
        radius_kind="NOMINAL_REFERENCE",
        radius=radius,
        artifact_ids=(SOURCE_ID,),
    )
    return gm, mass, radius, body


class SolarSystemSchemaTests(unittest.TestCase):
    def test_package_is_unpublished_and_public_rosters_are_exact(self):
        import jxplanetx.solar_system as package

        self.assertIs(type(package.__all__), list)
        self.assertEqual(package.__all__, [])
        expected_contract_exports = [
            "ArtifactBinding",
            "BodyParameter",
            "CoordinateEpoch",
            "CoverageInterval",
            "EphemerisProviderSpec",
            "EphemerisQuerySpec",
            "ExactUnitScale",
            "FrameRealization",
            "MAXIMUM_ARTIFACT_COUNT",
            "MAXIMUM_BODY_COUNT",
            "MAXIMUM_COVERAGE_INTERVAL_COUNT",
            "PhysicalConstant",
            "ProviderIdentity",
            "SolarSystemContractError",
            "SolarSystemCoverageError",
            "SolarSystemDataError",
            "SolarSystemDependencyUnavailableError",
            "UnitSystemDefinition",
            "validate_integrity",
        ]
        self.assertEqual(contracts_module.__all__, expected_contract_exports)
        self.assertEqual(contracts_module.__all__, sorted(contracts_module.__all__))
        self.assertEqual(
            serialization_module.__all__, sorted(serialization_module.__all__)
        )

    def test_exact_dataclass_field_rosters(self):
        rosters = {
            CoordinateEpoch: (
                "time_scale",
                "representation",
                "whole",
                "fraction",
                "coordinate_unit_id",
                "origin_id",
                "realization_id",
                "artifact_ids",
                "content_sha256",
            ),
            CoverageInterval: ("start", "end", "endpoint_policy", "content_sha256"),
            ArtifactBinding: (
                "artifact_id",
                "artifact_role",
                "provider_id",
                "version",
                "logical_locator",
                "locator_kind",
                "byte_length",
                "artifact_sha256",
                "media_type",
                "coverage_status",
                "coverage",
                "license_evidence_status",
                "license_spdx",
                "license_artifact_id",
                "redistribution_status",
                "load_order_status",
                "load_order",
                "extrapolation_policy",
                "content_integrity_class",
                "content_sha256",
            ),
            ProviderIdentity: (
                "provider_id",
                "implementation_id",
                "provider_kind",
                "version",
                "runtime_id",
                "module_name",
                "distribution_name",
                "distribution_version",
                "ordered_implementation_artifact_ids",
                "content_sha256",
            ),
            FrameRealization: (
                "frame_id",
                "frame_kind",
                "origin_kind",
                "origin_naif_id",
                "origin_realization_id",
                "axes_realization_id",
                "orientation_model_id",
                "orientation_time_dependence",
                "coordinate_time_scale",
                "coverage_status",
                "coverage",
                "artifact_ids",
                "content_sha256",
            ),
            ExactUnitScale: (
                "unit_id",
                "dimension",
                "si_unit_id",
                "numerator",
                "denominator",
                "definition_classification",
                "artifact_ids",
                "content_sha256",
            ),
            UnitSystemDefinition: (
                "unit_system_id",
                "length",
                "time",
                "mass",
                "gravitational_parameter",
                "coordinate_time_scale",
                "content_sha256",
            ),
            PhysicalConstant: (
                "constant_id",
                "quantity_kind",
                "source_decimal",
                "operational_value",
                "unit",
                "classification",
                "uncertainty_status",
                "uncertainty",
                "uncertainty_kind",
                "uncertainty_scale",
                "confidence_level",
                "covariance_group",
                "coordinate_scale_applicability",
                "coverage_status",
                "coverage",
                "artifact_ids",
                "content_sha256",
            ),
            BodyParameter: (
                "body_id",
                "naif_id",
                "body_role",
                "gravitational_parameter_status",
                "gravitational_parameter",
                "mass_status",
                "mass",
                "radius_status",
                "radius_kind",
                "radius",
                "artifact_ids",
                "content_sha256",
            ),
            EphemerisProviderSpec: (
                "spec_id",
                "identity",
                "artifacts",
                "ordered_load_artifact_ids",
                "capabilities",
                "native_time_scale",
                "native_frame",
                "native_output_unit_system",
                "network_access",
                "fallback_allowed",
                "extrapolation_policy",
                "evidence_class",
                "registry_authorized",
                "qualification_authorized",
                "content_sha256",
            ),
            EphemerisQuerySpec: (
                "query_id",
                "provider",
                "epoch",
                "target_naif_ids",
                "observer_naif_id",
                "frame",
                "aberration_correction",
                "output_unit_system",
                "state_kind",
                "target_chain_availability_status",
                "content_sha256",
            ),
        }
        for contract_type, expected in rosters.items():
            self.assertEqual(
                tuple(field.name for field in dataclasses.fields(contract_type)),
                expected,
            )


class SolarSystemContractConstructionTests(unittest.TestCase):
    def test_full_declarative_provider_query_and_body_context(self):
        provider, query, coverage, units = spk_context()
        gm, mass, radius, body = constants_and_body()
        for item in (
            provider,
            query,
            coverage,
            units,
            gm,
            mass,
            radius,
            body,
        ):
            validate_integrity(item)
            self.assertRegex(item.content_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(query.frame.content_sha256, provider.native_frame.content_sha256)
        self.assertEqual(
            query.output_unit_system.content_sha256,
            provider.native_output_unit_system.content_sha256,
        )
        self.assertEqual(body.gravitational_parameter, gm)
        self.assertEqual(body.mass, mass)
        self.assertEqual(body.radius, radius)

    def test_synthetic_offset_is_not_julian_date(self):
        epoch = CoordinateEpoch(
            "SYNTHETIC",
            "SYNTHETIC_OFFSET",
            4,
            -0.25,
            "unit.fixture.duration",
            "fixture.origin",
            "fixture.realization",
            (),
        )
        validate_integrity(epoch)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(epoch, coordinate_unit_id=DAY_UNIT, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(epoch, origin_id="JULIAN_DATE", content_sha256="")

    def test_spk_native_et_seconds_are_distinct_from_julian_day_labels(self):
        provider, query, _, _ = spk_context()
        self.assertEqual(
            query.epoch.representation,
            "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART",
        )
        self.assertEqual(query.epoch.coordinate_unit_id, SECOND_UNIT)
        self.assertEqual(query.epoch.origin_id, "SPICE_J2000_TDB_ORIGIN")
        self.assertEqual((query.epoch.whole, query.epoch.fraction), (0, 0.25))
        jd = physical_epoch(2_451_545, 0.0)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(query, epoch=jd, content_sha256="")
        jd_envelope = CoverageInterval(
            physical_epoch(2_451_544, 0.0),
            physical_epoch(2_451_546, 0.0),
            "CLOSED_CLOSED",
        )
        spk = provider.artifacts[0]
        mislabeled = dataclasses.replace(spk, coverage=(jd_envelope,), content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                provider,
                artifacts=(mislabeled, *provider.artifacts[1:]),
                content_sha256="",
            )

    def test_coverage_uses_exact_context_and_fraction_order(self):
        start = physical_epoch(2_451_545, 0.25)
        end = physical_epoch(2_451_545, 0.375)
        CoverageInterval(start, end, "CLOSED_OPEN")
        with self.assertRaises(SolarSystemContractError):
            CoverageInterval(end, start, "CLOSED_CLOSED")
        with self.assertRaises(SolarSystemContractError):
            CoverageInterval(start, dataclasses.replace(end, realization_id="other", content_sha256=""), "CLOSED_CLOSED")
        with self.assertRaises(SolarSystemContractError):
            CoverageInterval(start, start, "OPEN_OPEN")
        with self.assertRaises(SolarSystemContractError):
            CoverageInterval(
                start,
                dataclasses.replace(end, artifact_ids=(), content_sha256=""),
                "CLOSED_CLOSED",
            )

    def test_unit_system_proves_exact_gm_scale(self):
        length = ExactUnitScale("u.l", "LENGTH", "si.metre", 2, 1, "DEFINED_EXACT", ())
        time = ExactUnitScale("u.t", "TIME", "si.second", 3, 1, "DEFINED_EXACT", ())
        mass = ExactUnitScale("u.m", "MASS", "si.kilogram", 5, 1, "DEFINED_EXACT", ())
        gm = ExactUnitScale(
            "u.gm",
            "GRAVITATIONAL_PARAMETER",
            "si.metre3-per-second2",
            8,
            9,
            "DEFINED_EXACT",
            (),
        )
        UnitSystemDefinition("u.system", length, time, mass, gm, "SYNTHETIC")
        wrong = dataclasses.replace(gm, numerator=1, denominator=1, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            UnitSystemDefinition("u.bad", length, time, mass, wrong, "SYNTHETIC")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(gm, numerator=16, denominator=18, content_sha256="")
        nominal_length = dataclasses.replace(
            length,
            definition_classification="NOMINAL_EXACT",
            artifact_ids=(SOURCE_ID,),
            content_sha256="",
        )
        nominal_gm = dataclasses.replace(
            gm,
            definition_classification="NOMINAL_EXACT",
            artifact_ids=(SOURCE_ID,),
            content_sha256="",
        )
        UnitSystemDefinition(
            "u.nominal", nominal_length, time, mass, nominal_gm, "SYNTHETIC"
        )
        with self.assertRaises(SolarSystemContractError):
            UnitSystemDefinition("u.bad-class", nominal_length, time, mass, gm, "SYNTHETIC")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(length, si_unit_id="si.kilogram", content_sha256="")

    def test_decimal_is_authoritative_for_operational_binary64(self):
        exact = PhysicalConstant(
            "constant.day",
            "TIME",
            "86400",
            86_400.0,
            exact_units(scale="TDB", artifact_ids=(SOURCE_ID,)).time,
            "DEFINED_EXACT",
            "EXACT_NOT_APPLICABLE",
            None,
            None,
            "NOT_APPLICABLE",
            None,
            None,
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            None,
            (SOURCE_ID,),
        )
        validate_integrity(exact)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(exact, operational_value=86_401.0, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(exact, source_decimal="086400", content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(exact, source_decimal="86400.0", content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(exact, source_decimal="8.64e4", content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(exact, source_decimal="-0", operational_value=-0.0, content_sha256="")
        tiny_nonzero = "0." + "0" * 253 + "1"
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                exact,
                source_decimal=tiny_nonzero,
                operational_value=0.0,
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                exact,
                classification="FITTED",
                uncertainty_status="EXACT_NOT_APPLICABLE",
                content_sha256="",
            )

    def test_explicit_absence_statuses_are_cross_bound(self):
        source = external_artifact(artifact_id=SOURCE_ID, role="SOFTWARE_SOURCE")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(source, coverage=(spk_context()[2],), content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(source, load_order=0, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(source, license_artifact_id=None, content_sha256="")

    def test_massless_and_point_mass_absence_are_explicit(self):
        massless = BodyParameter(
            "TRACER",
            -1_000_001,
            "MASSLESS_TEST_PARTICLE",
            "NOT_PROVIDED_MASSLESS_TARGET",
            None,
            "NOT_PROVIDED_NOT_USED",
            None,
            "POINT_MASS_NO_COLLISION_RADIUS",
            None,
            None,
            (SOURCE_ID,),
        )
        validate_integrity(massless)
        gm = constants_and_body()[0]
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                massless,
                gravitational_parameter_status="PROVIDED_DYNAMICS_PARAMETER",
                gravitational_parameter=gm,
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(massless, radius_kind="MEAN", content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                massless,
                mass_status="PROVIDED_NOT_USED_BY_GM_DYNAMICS",
                mass=constants_and_body()[1],
                content_sha256="",
            )

    def test_physical_parameter_classification_and_source_status_are_closed(self):
        gm, mass, radius, body = constants_and_body()
        exact_gm = dataclasses.replace(
            gm,
            classification="NOMINAL_EXACT",
            uncertainty_status="EXACT_NOT_APPLICABLE",
            content_sha256="",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                body, gravitational_parameter=exact_gm, content_sha256=""
            )
        blocked_gm = dataclasses.replace(
            gm, coverage_status="NOT_RETAINED_BLOCKED", content_sha256=""
        )
        blocked_mass = dataclasses.replace(
            mass, coverage_status="NOT_RETAINED_BLOCKED", content_sha256=""
        )
        blocked_radius = dataclasses.replace(
            radius, coverage_status="NOT_RETAINED_BLOCKED", content_sha256=""
        )
        for field_name, blocked in (
            ("gravitational_parameter", blocked_gm),
            ("mass", blocked_mass),
            ("radius", blocked_radius),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(body, **{field_name: blocked, "content_sha256": ""})
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(body, radius_kind="MEAN", content_sha256="")
        barycenter = dataclasses.replace(
            body,
            naif_id=3,
            body_role="PLANETARY_SYSTEM_BARYCENTER",
            radius_status="POINT_MASS_NO_COLLISION_RADIUS",
            radius_kind=None,
            radius=None,
            content_sha256="",
        )
        validate_integrity(barycenter)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                barycenter,
                radius_status="PROVIDED_REFERENCE_RADIUS",
                radius_kind="NOMINAL_REFERENCE",
                radius=radius,
                content_sha256="",
            )

    def test_uncertainty_is_absolute_same_unit_and_confidence_is_symmetric(self):
        mass = constants_and_body()[1]
        confidence = dataclasses.replace(
            mass,
            uncertainty_kind="SYMMETRIC_CONFIDENCE_HALF_WIDTH",
            confidence_level=0.95,
            content_sha256="",
        )
        validate_integrity(confidence)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                confidence, uncertainty_scale="NOT_APPLICABLE", content_sha256=""
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(confidence, confidence_level=1.0, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                confidence,
                uncertainty_kind="STANDARD_UNCERTAINTY",
                content_sha256="",
            )

    def test_synthetic_provider_and_model_barycenter_are_distinct_contexts(self):
        provider, query = synthetic_context()
        validate_integrity(provider)
        validate_integrity(query)
        self.assertIsNone(query.observer_naif_id)
        self.assertTrue(all(target < 0 for target in query.target_naif_ids))
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(query, target_naif_ids=(10,), content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(query, observer_naif_id=0, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                query,
                epoch=dataclasses.replace(
                    query.epoch,
                    origin_id="different.synthetic.zero",
                    content_sha256="",
                ),
                content_sha256="",
            )
        blocked_frame = dataclasses.replace(
            provider.native_frame,
            coverage_status="NOT_RETAINED_BLOCKED",
            content_sha256="",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                provider, native_frame=blocked_frame, content_sha256=""
            )
        model_frame = FrameRealization(
            "frame.fixture.model-barycenter",
            "INERTIAL_ORIGIN_CENTERED",
            "NEWTONIAN_MODEL_BARYCENTER",
            None,
            "fixture.model-barycenter.from-selected-gm-roster",
            "fixture.icrf-aligned",
            "fixture.static-orientation",
            "STATIC",
            "TDB",
            "NOT_APPLICABLE",
            None,
            (),
        )
        validate_integrity(model_frame)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(model_frame, origin_naif_id=0, content_sha256="")

    def test_query_rejects_non_native_context_and_out_of_coverage(self):
        provider, query, _, units = spk_context()
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(query, aberration_correction="LT", content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(query, target_naif_ids=(10, 10), content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(query, observer_naif_id=10, content_sha256="")
        other_units = dataclasses.replace(units, unit_system_id="other", content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(query, output_unit_system=other_units, content_sha256="")
        with self.assertRaises(SolarSystemCoverageError):
            dataclasses.replace(query, epoch=spk_epoch(4_752_000, 0.0), content_sha256="")
        # Coarse artifact time containment is necessary only.  Exact target/
        # observer segment-chain availability remains a runtime obligation.
        unvalidated_target = dataclasses.replace(
            query, target_naif_ids=(999_999,), content_sha256=""
        )
        self.assertEqual(
            unvalidated_target.target_chain_availability_status,
            "REQUIRES_RUNTIME_PROVIDER_TARGET_AVAILABILITY_VALIDATION",
        )
        # Containment uses the represented coordinate context.  Optional
        # time-realization evidence may differ from the coarse bound evidence.
        independently_evidenced_epoch = spk_epoch(
            0, 0.25, artifact_ids=(SOURCE_ID,)
        )
        validate_integrity(
            dataclasses.replace(
                query,
                epoch=independently_evidenced_epoch,
                content_sha256="",
            )
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                query,
                epoch=spk_epoch(0, 0.25, artifact_ids=(SPK_ID,)),
                content_sha256="",
            )
        self.assertIs(provider, query.provider)


class SolarSystemIntegrityAndMutationTests(unittest.TestCase):
    def test_stale_outer_and_nested_seals_reject(self):
        _, query, _, _ = spk_context()
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(query, query_id="changed")
        object.__setattr__(query.epoch, "whole", query.epoch.whole + 1)
        with self.assertRaises(SolarSystemContractError):
            validate_integrity(query)

    def test_bool_subclasses_and_exact_tuple_types_reject(self):
        class TextSubclass(str):
            pass

        with self.assertRaises(SolarSystemContractError):
            CoordinateEpoch("TDB", "JD_TWO_PART", True, 0.0, DAY_UNIT, "JULIAN_DATE", "r", ())
        with self.assertRaises(SolarSystemContractError):
            CoordinateEpoch("TDB", "JD_TWO_PART", 1, 0, DAY_UNIT, "JULIAN_DATE", "r", ())
        with self.assertRaises(SolarSystemContractError):
            CoordinateEpoch(
                TextSubclass("TDB"),
                "JD_TWO_PART",
                1,
                0.0,
                DAY_UNIT,
                "JULIAN_DATE",
                "r",
                (),
            )
        with self.assertRaises(SolarSystemContractError):
            CoordinateEpoch("TDB", "JD_TWO_PART", 1, 0.0, DAY_UNIT, "JULIAN_DATE", "r", [])  # type: ignore[arg-type]

        class EpochSubclass(CoordinateEpoch):
            pass

        with self.assertRaises(SolarSystemContractError):
            EpochSubclass(
                "TDB", "JD_TWO_PART", 1, 0.0, DAY_UNIT, "JULIAN_DATE", "r", ()
            )

    def test_every_sequence_field_requires_an_exact_tuple(self):
        provider, query, coverage, units = spk_context()
        _, _, _, body = constants_and_body()
        cases = (
            (query.epoch, "artifact_ids", list(query.epoch.artifact_ids)),
            (coverage, "endpoint_policy", ["CLOSED_CLOSED"]),
            (provider.artifacts[0], "coverage", list(provider.artifacts[0].coverage or ())),
            (
                provider.identity,
                "ordered_implementation_artifact_ids",
                list(provider.identity.ordered_implementation_artifact_ids),
            ),
            (provider.native_frame, "artifact_ids", list(provider.native_frame.artifact_ids)),
            (units.length, "artifact_ids", list(units.length.artifact_ids)),
            (body, "artifact_ids", list(body.artifact_ids)),
            (provider, "artifacts", list(provider.artifacts)),
            (provider, "capabilities", list(provider.capabilities)),
            (query, "target_naif_ids", list(query.target_naif_ids)),
        )
        for item, field_name, replacement in cases:
            with self.subTest(type=type(item).__name__, field=field_name):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(
                        item,
                        **{field_name: replacement, "content_sha256": ""},
                    )

    def test_identifier_text_is_printable_ascii_and_set_rosters_are_sorted(self):
        epoch = physical_epoch(1, 0.0, artifact_ids=("a", "b"))
        for bad in ("embedded\nline", "nonascii-é", "control-\x1f"):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(epoch, realization_id=bad, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(epoch, artifact_ids=("b", "a"), content_sha256="")

    def test_provider_rosters_locators_and_local_custody_are_canonical(self):
        provider, _, _, _ = spk_context()
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                provider,
                artifacts=tuple(reversed(provider.artifacts)),
                content_sha256="",
            )
        source = provider.artifacts[-1]
        duplicate_locator_source = dataclasses.replace(
            source,
            logical_locator=provider.artifacts[1].logical_locator,
            content_sha256="",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                provider,
                artifacts=(
                    provider.artifacts[0],
                    provider.artifacts[1],
                    duplicate_locator_source,
                ),
                content_sha256="",
            )
        spk = provider.artifacts[0]
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                spk,
                locator_kind="EXTERNAL_REFERENCE_ONLY",
                logical_locator="https://example.invalid/kernel.bsp",
                content_sha256="",
            )
        bad_identity = dataclasses.replace(
            provider.identity,
            version="2",
            distribution_version="2",
            content_sha256="",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(provider, identity=bad_identity, content_sha256="")
        native_id = "artifact.provider.native"
        native = external_artifact(
            artifact_id=native_id,
            role="NATIVE_LIBRARY",
            version="N0067",
        )
        wrapper_plus_native_identity = dataclasses.replace(
            provider.identity,
            ordered_implementation_artifact_ids=(SOURCE_ID, native_id),
            content_sha256="",
        )
        wrapper_plus_native = dataclasses.replace(
            provider,
            identity=wrapper_plus_native_identity,
            artifacts=(
                provider.artifacts[0],
                provider.artifacts[1],
                native,
                provider.artifacts[2],
            ),
            content_sha256="",
        )
        validate_integrity(wrapper_plus_native)

    def test_fixed_tokens_and_serializer_fail_closed_without_protocol_probing(self):
        provider, query, _, _ = spk_context()

        class ComparisonBomb:
            def __ne__(self, other):
                raise RuntimeError("comparison protocol executed")

        for item, field_name in (
            (provider.artifacts[0], "extrapolation_policy"),
            (provider, "evidence_class"),
            (query, "state_kind"),
        ):
            with self.subTest(type=type(item).__name__, field=field_name):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(
                        item,
                        **{field_name: ComparisonBomb(), "content_sha256": ""},
                    )

        class DataclassProbeBombMeta(type):
            def __getattribute__(cls, name):
                if name == "__dataclass_fields__":
                    raise RuntimeError("dataclass protocol executed")
                return super().__getattribute__(name)

        class DataclassProbeBomb(metaclass=DataclassProbeBombMeta):
            pass

        with self.assertRaises(SolarSystemContractError):
            canonical_json(DataclassProbeBomb())

    def test_pretraversal_caps_and_wrong_items_fail_as_contract_errors(self):
        provider, query, _, _ = spk_context()
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(provider, artifacts=(object(),), content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                provider.native_frame,
                coverage=(object(),) * 17,
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                query.epoch,
                artifact_ids=tuple(f"artifact.{index:02d}" for index in range(65)),
                content_sha256="",
            )

    def test_negative_zero_epoch_is_rejected_but_serializer_preserves_sign(self):
        with self.assertRaises(SolarSystemContractError):
            physical_epoch(2_451_545, -0.0)
        positive = canonical_json(0.0)
        negative = canonical_json(-0.0)
        self.assertEqual(positive, b'{"float_hex":"0x0.0p+0"}')
        self.assertEqual(negative, b'{"float_hex":"-0x0.0p+0"}')
        self.assertNotEqual(positive, negative)

    def test_count_caps_precede_item_traversal(self):
        original_text = contracts_module._text

        def guarded_text(value, label, **kwargs):
            if type(value) is object:
                raise AssertionError("item traversal occurred before count rejection")
            return original_text(value, label, **kwargs)

        with mock.patch.object(contracts_module, "_text", side_effect=guarded_text):
            with self.assertRaises(SolarSystemContractError):
                ProviderIdentity(
                    "provider",
                    "implementation",
                    "SYNTHETIC_TEST",
                    "1",
                    "runtime",
                    "module",
                    "distribution",
                    "1",
                    (object(),) * 65,  # type: ignore[arg-type]
                )
        _, query, _, _ = spk_context()
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(query, target_naif_ids=(object(),) * 65, content_sha256="")

    def test_contract_text_cap_precedes_trim_traversal(self):
        original = contracts_module._is_trimmed_contract_text

        def guarded_trim(value):
            if len(value) > 256:
                raise AssertionError("trim traversal occurred before code-point cap")
            return original(value)

        with mock.patch.object(
            contracts_module,
            "_is_trimmed_contract_text",
            side_effect=guarded_trim,
        ):
            with self.assertRaises(SolarSystemContractError):
                contracts_module._text("x" * 257, "oversized")

    def test_every_contract_field_is_bound_by_the_self_excluding_digest(self):
        provider, query, coverage, units = spk_context()
        gm, _, _, body = constants_and_body()
        objects = (
            query.epoch,
            coverage,
            provider.artifacts[0],
            provider.identity,
            provider.native_frame,
            units.length,
            units,
            gm,
            body,
            provider,
            query,
        )
        for item in objects:
            slug = item._SLUG
            domain = f"jxplanetx.solar-system.{slug}.content-integrity.v1"
            schema = f"jxplanetx.solar-system.{slug}.payload.v1"
            pairs = tuple(
                (field.name, getattr(item, field.name))
                for field in dataclasses.fields(item)
                if field.name != "content_sha256"
            )
            baseline = domain_sha256(
                domain,
                schema,
                {
                    "dataclass": f"{type(item).__module__}.{type(item).__qualname__}",
                    "fields": pairs,
                },
            )
            self.assertEqual(baseline, item.content_sha256)
            for index, (name, value) in enumerate(pairs):
                changed = list(pairs)
                changed[index] = (name, (value, "FIELD_MUTATION"))
                mutated = domain_sha256(
                    domain,
                    schema,
                    {
                        "dataclass": f"{type(item).__module__}.{type(item).__qualname__}",
                        "fields": tuple(changed),
                    },
                )
                with self.subTest(type=type(item).__name__, field=name):
                    self.assertNotEqual(mutated, baseline)


class SolarSystemSerializationTests(unittest.TestCase):
    def test_mapping_order_and_sorted_json_roundtrip(self):
        value = {"z": (-0.0, "é"), "a": True}
        encoded = canonical_json(value)
        self.assertTrue(encoded.startswith(b'{"a":true,"z":'))
        loaded = json.loads(encoded.decode("utf-8"))
        rerendered = json.dumps(
            loaded,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        self.assertEqual(rerendered, encoded)
        query_encoded = canonical_json(spk_context()[1])
        query_loaded = json.loads(query_encoded.decode("ascii"))
        self.assertEqual(
            json.dumps(
                query_loaded,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii"),
            query_encoded,
        )

    def test_string_cap_and_cycle_reject_before_unbounded_encoding(self):
        with mock.patch.object(
            serialization_module.json,
            "dumps",
            side_effect=AssertionError("encoder called before string cap"),
        ):
            with self.assertRaises(SolarSystemContractError):
                canonical_json("x" * 4_097)
        cyclic: list[object] = []
        cyclic.append(cyclic)
        with self.assertRaises(SolarSystemContractError):
            canonical_json(cyclic)

    def test_serializer_count_depth_integer_and_byte_caps_are_hard(self):
        with self.assertRaises(SolarSystemContractError):
            canonical_json((None,) * 257)
        with self.assertRaises(SolarSystemContractError):
            canonical_json({f"k{index:03d}": None for index in range(257)})
        deep: object = None
        for _ in range(34):
            deep = (deep,)
        with self.assertRaises(SolarSystemContractError):
            canonical_json(deep)
        with self.assertRaises(SolarSystemContractError):
            canonical_json(1 << 4_096)
        encoded = canonical_json({"bounded": -0.0})
        with self.assertRaises(SolarSystemContractError):
            canonical_json({"bounded": -0.0}, maximum_bytes=len(encoded) - 1)

    def test_domain_and_schema_are_both_bound(self):
        value = {"field": 1.0}
        baseline = domain_sha256("domain.a", "schema.a", value)
        self.assertNotEqual(baseline, domain_sha256("domain.b", "schema.a", value))
        self.assertNotEqual(baseline, domain_sha256("domain.a", "schema.b", value))
        self.assertNotEqual(baseline, domain_sha256("domain.a", "schema.a", {"field": -1.0}))

    def test_complete_literal_coordinate_epoch_production_preimage(self):
        epoch = CoordinateEpoch(
            "TDB",
            "JD_TWO_PART",
            2_451_545,
            0.0,
            DAY_UNIT,
            "JULIAN_DATE",
            "fixture.tdb-realization",
            (SOURCE_ID,),
        )
        payload = (
            b'{"dataclass":"jxplanetx.solar_system.contracts.CoordinateEpoch",'
            b'"fields":[["time_scale","TDB"],["representation","JD_TWO_PART"],'
            b'["whole",2451545],["fraction",{"float_hex":"0x0.0p+0"}],'
            b'["coordinate_unit_id","unit.day.86400-si-seconds.defined-exact.v1"],'
            b'["origin_id","JULIAN_DATE"],["realization_id","fixture.tdb-realization"],'
            b'["artifact_ids",["artifact.provider.source"]]]}'
        )
        preimage = (
            b"jxplanetx.solar-system.coordinate-epoch.content-integrity.v1\x00"
            b"jxplanetx.solar-system.coordinate-epoch.payload.v1\x00"
            + payload
        )
        self.assertEqual(len(preimage), 484)
        self.assertEqual(
            hashlib.sha256(preimage).hexdigest(),
            "51ffd7575e7c6c98d025d2a2d4e8759b6c33ed96d99b401251cf1abb0d30d8da",
        )
        self.assertEqual(epoch.content_sha256, hashlib.sha256(preimage).hexdigest())

    def test_complete_literal_provider_identity_production_preimage(self):
        identity = provider_identity()
        payload = (
            b'{"dataclass":"jxplanetx.solar_system.contracts.ProviderIdentity",'
            b'"fields":[["provider_id","provider.fixture"],'
            b'["implementation_id","provider.fixture.implementation.v1"],'
            b'["provider_kind","LOCAL_OFFLINE_SPK"],["version","1"],'
            b'["runtime_id","cpython.test"],["module_name","fixture_provider"],'
            b'["distribution_name","fixture-provider"],'
            b'["distribution_version","1"],'
            b'["ordered_implementation_artifact_ids",["artifact.provider.source"]]]}'
        )
        preimage = (
            b"jxplanetx.solar-system.provider-identity.content-integrity.v1\x00"
            b"jxplanetx.solar-system.provider-identity.payload.v1\x00"
            + payload
        )
        self.assertEqual(len(preimage), 542)
        self.assertEqual(
            hashlib.sha256(preimage).hexdigest(),
            "14d31c785f7d8e81b8925f110a9751fe2b3c98c2a5394730b9934d53dcd4f96f",
        )
        self.assertEqual(identity.content_sha256, hashlib.sha256(preimage).hexdigest())
        self.assertNotEqual(
            hashlib.sha256(preimage.replace(b"LOCAL_OFFLINE_SPK", b"SYNTHETIC_TEST")).hexdigest(),
            identity.content_sha256,
        )

    def test_all_production_domains_have_locked_fixture_digests(self):
        provider, query, coverage, units = spk_context()
        gm, mass, radius, body = constants_and_body()
        objects = (
            query.epoch,
            coverage,
            provider.artifacts[0],
            provider.identity,
            provider.native_frame,
            units.length,
            units,
            gm,
            body,
            provider,
            query,
        )
        expected = (
            "9ef1680f79e0750bf954b852a560a785df615e8c4f89879b4316623d6706d8c9",
            "e79a17550b0de0add253baede0e2d2130bd82a3e09594a92811bab435148778e",
            "2d807570d392b88f4d5cd75dcf8003efb07be27e952d1e58b36bcadd549ec09c",
            "14d31c785f7d8e81b8925f110a9751fe2b3c98c2a5394730b9934d53dcd4f96f",
            "6c7423d46bc61bb9ac8af696478f27afee02587526223b11e29282e794173642",
            "d5cf34b679967b9621f96665bbdec49d2b985b5b02e099b321f506a866d6e6fc",
            "23285469de819bb20a8a4e852122a9a9d3f3ba0aa3b6b24f78dfb17942a36f40",
            "b1f0b365d48e8e46806f544d0985146a7cc5e43fcaeda10d73c6dbb31a6b0f0f",
            "c4a7913315d360169f02c2379fa27d69f2407ae54730a14a26377a0c200198fb",
            "3f754b7fc97535c2b15c91b749f7ac706551249112b985788cb002ad017fd94b",
            "3ed9e94d4368b1bd80d87dcce66e3fdd457c29cf185d5502c104873706049b17",
        )
        self.assertEqual(tuple(item.content_sha256 for item in objects), expected)
        self.assertNotEqual(gm.content_sha256, mass.content_sha256)
        self.assertNotEqual(mass.content_sha256, radius.content_sha256)


class SolarSystemDependencyBoundaryTests(unittest.TestCase):
    def test_clean_package_and_contract_imports_load_no_astronomy_dependency(self):
        root = Path(__file__).resolve().parents[1]
        script = """
import sys
import jxplanetx.solar_system as package
assert package.__all__ == []
import jxplanetx.solar_system.serialization
import jxplanetx.solar_system.contracts
for name in ('numpy', 'astropy', 'erfa', 'spiceypy', 'jplephem', 'skyfield'):
    assert name not in sys.modules, name
"""
        environment = dict(os.environ)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONPATH"] = str(root / "src")
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        # -I ignores PYTHONPATH, so repeat without isolation if the source tree
        # is not installed in the selected runtime.
        if completed.returncode != 0 and "No module named 'jxplanetx'" in completed.stderr:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
