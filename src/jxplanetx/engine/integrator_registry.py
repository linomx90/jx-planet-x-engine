"""Inspectable execution registry for unified JX integrator routes.

The registry describes only routes dispatched by :class:`JXSimulation`.  It
does not promote the broader capability catalog to an execution surface and
does not authorize scientific or production claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from jxplanetx.gr15 import GR15Spec

from .contracts import (
    CannonballSRP,
    MutualEIH1PN,
    NewtonianPointMass,
    RestrictedStaticCentral1PN,
)
from .trajectory_contracts import ADAPTIVE_RKF78_METHOD_ID, AdaptiveRKF78Spec


INTEGRATOR_ROUTE_REGISTRY_VERSION = 1
INTEGRATOR_ROUTE_SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"

FORCE_ABI_V1_GENERAL_FIRST_ORDER_SCOPE = (
    "FORCE_ABI_V1_GENERAL_FIRST_ORDER_ORDERED_PLAN"
)
EXACT_ORDERED_FORCE_MODEL_IDS_SCOPE = "EXACT_ORDERED_FORCE_MODEL_IDS"

GR15_NEWTONIAN_METHOD_ID = "JX_GAUSS_RADAU15_V3"
GR15_EIH1PN_METHOD_ID = "JX_GR15_EIH1PN_V1"
GR15_EIH1PN_CUDA_METHOD_ID = "JX_GR15_EIH1PN_CUDA_V1"


@dataclass(frozen=True, slots=True)
class IntegratorRouteSpec:
    """Immutable compatibility and claim boundary for one executable route."""

    integrator_id: str
    spec_type: type
    supported_backend_ids: tuple[str, ...]
    force_plan_scope: str
    required_force_model_ids: tuple[str, ...]
    required_force_abi_integrator_class: str | None
    continuation_backend_ids: tuple[str, ...] = ()
    archive_backend_ids: tuple[str, ...] = ()
    portable_manifest_backend_ids: tuple[str, ...] = ()
    portable_manifest_force_model_ids: tuple[str, ...] = ()
    scientific_claim_state: str = INTEGRATOR_ROUTE_SCIENTIFIC_CLAIM_STATE
    production_authorized: bool = False
    qualification_authorized: bool = False
    general_superiority_claimed: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.integrator_id) is not str
            or not self.integrator_id
            or self.integrator_id.strip() != self.integrator_id
        ):
            raise ValueError("integrator_id must be a nonempty, trimmed string")
        if type(self.spec_type) is not type:
            raise TypeError("spec_type must be an exact type")
        if (
            type(self.supported_backend_ids) is not tuple
            or not self.supported_backend_ids
            or any(
                type(value) is not str or not value
                for value in self.supported_backend_ids
            )
            or len(set(self.supported_backend_ids)) != len(self.supported_backend_ids)
        ):
            raise ValueError("supported_backend_ids must be unique nonempty strings")
        if self.force_plan_scope not in {
            FORCE_ABI_V1_GENERAL_FIRST_ORDER_SCOPE,
            EXACT_ORDERED_FORCE_MODEL_IDS_SCOPE,
        }:
            raise ValueError("force_plan_scope is not a registered route scope")
        if (
            type(self.required_force_model_ids) is not tuple
            or any(
                type(value) is not str or not value
                for value in self.required_force_model_ids
            )
            or len(set(self.required_force_model_ids))
            != len(self.required_force_model_ids)
        ):
            raise ValueError("required_force_model_ids must be unique strings")
        if (
            self.force_plan_scope == EXACT_ORDERED_FORCE_MODEL_IDS_SCOPE
            and not self.required_force_model_ids
        ):
            raise ValueError("an exact force-plan scope requires model identifiers")
        if (
            self.force_plan_scope == FORCE_ABI_V1_GENERAL_FIRST_ORDER_SCOPE
            and self.required_force_model_ids
        ):
            raise ValueError("the general force-plan scope cannot require one roster")
        if self.force_plan_scope == FORCE_ABI_V1_GENERAL_FIRST_ORDER_SCOPE:
            if self.required_force_abi_integrator_class != "GENERAL_FIRST_ORDER":
                raise ValueError(
                    "the general force-plan scope requires GENERAL_FIRST_ORDER"
                )
        elif self.required_force_abi_integrator_class is not None:
            raise ValueError(
                "specialized exact-plan routes are not force-ABI-dispatched"
            )

        capability_rows = (
            ("continuation", self.continuation_backend_ids),
            ("archive", self.archive_backend_ids),
            ("portable_manifest", self.portable_manifest_backend_ids),
        )
        supported = set(self.supported_backend_ids)
        for label, backend_ids in capability_rows:
            if (
                type(backend_ids) is not tuple
                or any(type(value) is not str or not value for value in backend_ids)
                or len(set(backend_ids)) != len(backend_ids)
                or not set(backend_ids).issubset(supported)
            ):
                raise ValueError(
                    f"{label}_backend_ids must be unique supported backends"
                )
        if (
            type(self.portable_manifest_force_model_ids) is not tuple
            or any(
                type(value) is not str or not value
                for value in self.portable_manifest_force_model_ids
            )
            or len(set(self.portable_manifest_force_model_ids))
            != len(self.portable_manifest_force_model_ids)
        ):
            raise ValueError(
                "portable_manifest_force_model_ids must be unique strings"
            )
        if bool(self.portable_manifest_backend_ids) != bool(
            self.portable_manifest_force_model_ids
        ):
            raise ValueError(
                "portable manifest backends and force-model roster must be declared together"
            )

        if (
            self.scientific_claim_state
            != INTEGRATOR_ROUTE_SCIENTIFIC_CLAIM_STATE
            or type(self.production_authorized) is not bool
            or self.production_authorized
            or type(self.qualification_authorized) is not bool
            or self.qualification_authorized
            or type(self.general_superiority_claimed) is not bool
            or self.general_superiority_claimed
        ):
            raise ValueError(
                "integrator routes remain screening-only, unqualified, and "
                "nonproduction"
            )

    def supports_backend(self, backend_id: object, device: object) -> bool:
        """Return whether an exact backend/device request belongs to this route."""

        if type(backend_id) is not str or type(device) is not str:
            return False
        if backend_id not in self.supported_backend_ids:
            return False
        if backend_id == "numpy":
            return device == "cpu"
        if backend_id != "cupy":
            return False
        if device in {"cuda", "gpu"}:
            return True
        prefix, separator, suffix = device.partition(":")
        return (
            separator == ":"
            and prefix in {"cuda", "gpu"}
            and bool(suffix)
            and suffix.isdigit()
        )

    def supports_continuation(self, backend_id: object, device: object) -> bool:
        return (
            self.supports_backend(backend_id, device)
            and backend_id in self.continuation_backend_ids
        )

    def supports_archive(self, backend_id: object, device: object) -> bool:
        return (
            self.supports_backend(backend_id, device)
            and backend_id in self.archive_backend_ids
        )

    def supports_portable_manifest(
        self,
        backend_id: object,
        device: object,
        force_model_ids: object,
    ) -> bool:
        if (
            not self.supports_backend(backend_id, device)
            or backend_id not in self.portable_manifest_backend_ids
            or type(force_model_ids) is not tuple
            or not force_model_ids
            or any(type(value) is not str or not value for value in force_model_ids)
        ):
            return False
        allowed = set(self.portable_manifest_force_model_ids)
        return all(value in allowed for value in force_model_ids)


_ROUTES = (
    IntegratorRouteSpec(
        integrator_id=ADAPTIVE_RKF78_METHOD_ID,
        spec_type=AdaptiveRKF78Spec,
        supported_backend_ids=("numpy", "cupy"),
        force_plan_scope=FORCE_ABI_V1_GENERAL_FIRST_ORDER_SCOPE,
        required_force_model_ids=(),
        required_force_abi_integrator_class="GENERAL_FIRST_ORDER",
        continuation_backend_ids=("numpy",),
        archive_backend_ids=("numpy",),
        portable_manifest_backend_ids=("numpy",),
        portable_manifest_force_model_ids=(
            NewtonianPointMass.MODEL_ID,
            RestrictedStaticCentral1PN.MODEL_ID,
            CannonballSRP.MODEL_ID,
        ),
    ),
    IntegratorRouteSpec(
        integrator_id=GR15_NEWTONIAN_METHOD_ID,
        spec_type=GR15Spec,
        supported_backend_ids=("numpy",),
        force_plan_scope=EXACT_ORDERED_FORCE_MODEL_IDS_SCOPE,
        required_force_model_ids=(NewtonianPointMass.MODEL_ID,),
        required_force_abi_integrator_class=None,
        continuation_backend_ids=("numpy",),
        archive_backend_ids=("numpy",),
    ),
    IntegratorRouteSpec(
        integrator_id=GR15_EIH1PN_METHOD_ID,
        spec_type=GR15Spec,
        supported_backend_ids=("numpy",),
        force_plan_scope=EXACT_ORDERED_FORCE_MODEL_IDS_SCOPE,
        required_force_model_ids=(
            NewtonianPointMass.MODEL_ID,
            MutualEIH1PN.MODEL_ID,
        ),
        required_force_abi_integrator_class=None,
        continuation_backend_ids=("numpy",),
        archive_backend_ids=("numpy",),
    ),
    IntegratorRouteSpec(
        integrator_id=GR15_EIH1PN_CUDA_METHOD_ID,
        spec_type=GR15Spec,
        supported_backend_ids=("cupy",),
        force_plan_scope=EXACT_ORDERED_FORCE_MODEL_IDS_SCOPE,
        required_force_model_ids=(
            NewtonianPointMass.MODEL_ID,
            MutualEIH1PN.MODEL_ID,
        ),
        required_force_abi_integrator_class=None,
    ),
)

INTEGRATOR_ROUTE_REGISTRY: Mapping[str, IntegratorRouteSpec] = MappingProxyType(
    {route.integrator_id: route for route in _ROUTES}
)


def list_integrator_routes() -> tuple[IntegratorRouteSpec, ...]:
    """Return executable routes in stable registration order."""

    return _ROUTES


def get_integrator_route(integrator_id: str) -> IntegratorRouteSpec:
    """Resolve one exact unified-simulation route identifier."""

    if (
        type(integrator_id) is not str
        or not integrator_id
        or integrator_id.strip() != integrator_id
    ):
        raise KeyError("integrator_id must be a nonempty, trimmed string")
    try:
        return INTEGRATOR_ROUTE_REGISTRY[integrator_id]
    except KeyError as exc:
        raise KeyError(f"unknown JX integrator route {integrator_id!r}") from exc


__all__ = [
    "EXACT_ORDERED_FORCE_MODEL_IDS_SCOPE",
    "FORCE_ABI_V1_GENERAL_FIRST_ORDER_SCOPE",
    "GR15_EIH1PN_CUDA_METHOD_ID",
    "GR15_EIH1PN_METHOD_ID",
    "GR15_NEWTONIAN_METHOD_ID",
    "INTEGRATOR_ROUTE_REGISTRY",
    "INTEGRATOR_ROUTE_REGISTRY_VERSION",
    "INTEGRATOR_ROUTE_SCIENTIFIC_CLAIM_STATE",
    "IntegratorRouteSpec",
    "get_integrator_route",
    "list_integrator_routes",
]
