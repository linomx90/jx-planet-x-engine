"""Fail-closed relationships between already-declared frame contracts.

This unpublished milestone classifies exact ``FrameRealization`` declarations;
it does not transform coordinates, execute an ephemeris provider, parse a
kernel, inspect arrays, or establish that a named realization is authentic.
The returned tokens are diagnostics, not executable operation plans.

In particular, a declared SPICE-J2000/ICRF-aligned barycentric frame remains a
TDB-compatible Solar-System-barycentric inertial context, not formal TCB BCRS.
An Earth-centered inertial origin is not thereby GCRS, ITRS, body-fixed, or an
ecliptic frame.  A different-origin diagnostic neither proves that a provider
can supply the required origin states nor specifies a translation direction.
It does not establish that a translation exists, is executable, or is
sufficient.  Time-dependent orientation and every numerical frame operation
remain outside this module.
"""

from __future__ import annotations

from .contracts import (
    FrameRealization,
    SolarSystemContractError,
    validate_integrity,
)


# This tuple is intentionally ordered by classifier decision precedence rather
# than lexicographically.  The public module roster itself remains sorted.
FRAME_RELATION_TOKENS = (
    "EXACT_FRAME_CONTRACT_IDENTITY",
    "SAME_DECLARED_COORDINATE_MAP_DIFFERENT_CONTRACT_CUSTODY_UNSUPPORTED",
    "UNRESOLVED_FRAME_ALIAS",
    "SAME_DECLARED_AXES_AND_COORDINATE_CONTEXT_DIFFERENT_ORIGIN_TRANSLATION_REQUIRED_UNSUPPORTED",
    "AXES_OR_COORDINATE_CONTEXT_TRANSFORMATION_REQUIRED_UNSUPPORTED",
)

_EXACT_IDENTITY = FRAME_RELATION_TOKENS[0]
_SAME_MAP_DIFFERENT_CUSTODY = FRAME_RELATION_TOKENS[1]
_UNRESOLVED_ALIAS = FRAME_RELATION_TOKENS[2]
_DIFFERENT_ORIGIN = FRAME_RELATION_TOKENS[3]
_AXES_OR_CONTEXT = FRAME_RELATION_TOKENS[4]


def _require_frame(value: object, label: str) -> FrameRealization:
    if type(value) is not FrameRealization:
        raise SolarSystemContractError(f"{label} must be an exact FrameRealization")
    validate_integrity(value)
    return value


def _axes_and_coordinate_context_key(
    value: FrameRealization,
) -> tuple[str, str, str, str, str]:
    return (
        value.frame_kind,
        value.axes_realization_id,
        value.orientation_model_id,
        value.orientation_time_dependence,
        value.coordinate_time_scale,
    )


def _origin_key(value: FrameRealization) -> tuple[str, int | None, str]:
    return (
        value.origin_kind,
        value.origin_naif_id,
        value.origin_realization_id,
    )


def _full_semantic_key(
    value: FrameRealization,
) -> tuple[tuple[str, str, str, str, str], tuple[str, int | None, str]]:
    return (_axes_and_coordinate_context_key(value), _origin_key(value))


def _supports_origin_only_diagnostic(
    source: FrameRealization,
    target: FrameRealization,
) -> bool:
    if (
        _origin_key(source) == _origin_key(target)
        or source.coordinate_time_scale != "TDB"
        or target.coordinate_time_scale != "TDB"
        or source.orientation_time_dependence != "STATIC"
        or target.orientation_time_dependence != "STATIC"
        or source.axes_realization_id != target.axes_realization_id
        or source.orientation_model_id != target.orientation_model_id
    ):
        return False

    physical_local_origins = frozenset(
        ("BODY_CENTER", "PLANETARY_SYSTEM_BARYCENTER")
    )
    if (
        source.frame_kind == "INERTIAL_ORIGIN_CENTERED"
        and target.frame_kind == "INERTIAL_ORIGIN_CENTERED"
    ):
        return (
            source.origin_kind in physical_local_origins
            and target.origin_kind in physical_local_origins
            and source.origin_naif_id is not None
            and target.origin_naif_id is not None
        )

    if frozenset((source.frame_kind, target.frame_kind)) != frozenset(
        (
            "INERTIAL_ORIGIN_CENTERED",
            "TDB_COMPATIBLE_BARYCENTRIC_INERTIAL",
        )
    ):
        return False

    barycentric = (
        source
        if source.frame_kind == "TDB_COMPATIBLE_BARYCENTRIC_INERTIAL"
        else target
    )
    local = target if barycentric is source else source
    return (
        barycentric.origin_kind == "SOLAR_SYSTEM_BARYCENTER"
        and barycentric.origin_naif_id == 0
        and local.origin_kind in physical_local_origins
        and local.origin_naif_id is not None
    )


def classify_frame_relation(
    source_frame: FrameRealization,
    target_frame: FrameRealization,
) -> str:
    """Classify two exact frame declarations without performing an operation.

    ``EXACT_FRAME_CONTRACT_IDENTITY`` requires the same frame identifier,
    declared coordinate-map semantics, and complete M1 content seal including
    coverage and artifact evidence.  Equal declared semantics under different
    identifiers remain an unresolved alias.  Evidence is validated separately
    and is never merged or treated as authenticity.
    """

    source = _require_frame(source_frame, "source_frame")
    target = _require_frame(target_frame, "target_frame")
    source_semantics = _full_semantic_key(source)
    target_semantics = _full_semantic_key(target)

    if source.frame_id == target.frame_id:
        if source_semantics != target_semantics:
            raise SolarSystemContractError(
                "one frame_id cannot name different declared coordinate-map semantics"
            )
        if source.content_sha256 == target.content_sha256:
            return _EXACT_IDENTITY
        return _SAME_MAP_DIFFERENT_CUSTODY

    if source_semantics == target_semantics:
        return _UNRESOLVED_ALIAS

    if _supports_origin_only_diagnostic(source, target):
        return _DIFFERENT_ORIGIN

    return _AXES_OR_CONTEXT


def require_exact_frame_contract_identity(
    source_frame: FrameRealization,
    target_frame: FrameRealization,
) -> None:
    """Require exact frame-contract identity or fail without partial output."""

    relation = classify_frame_relation(source_frame, target_frame)
    if relation != _EXACT_IDENTITY:
        raise SolarSystemContractError(
            "frame relation is not exact contract identity: " + relation
        )


__all__ = [
    "FRAME_RELATION_TOKENS",
    "classify_frame_relation",
    "require_exact_frame_contract_identity",
]
