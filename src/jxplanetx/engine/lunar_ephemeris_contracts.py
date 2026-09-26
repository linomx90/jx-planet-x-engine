"""Dependency-light identities for the public lunar ephemeris v1 route."""

LUNAR_EPHEMERIS_V1_MODEL_ID = (
    "solar-system.screening.resolved-eleven-eih1pn-coupled-lunar.v1"
)
LUNAR_EPHEMERIS_V1_METHOD_ID = "jx.integrator.rkf78.fixed.coupled-lunar.v1"


__all__ = [
    "LUNAR_EPHEMERIS_V1_METHOD_ID",
    "LUNAR_EPHEMERIS_V1_MODEL_ID",
]
