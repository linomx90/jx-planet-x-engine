"""Content-addressed long-horizon observed-object benchmark populations.

The nominal elements in this module are selected deterministically from
retained NASA/JPL SBDB responses at the exact frozen DE441 source epoch.  They
form an observed-object numerical benchmark, not an observationally complete
or debiased population model.  The perturbation ensembles use bounded,
independent offsets scaled by source-reported one-sigma element uncertainties;
because the full covariance matrices are not present, they are sensitivity
envelopes rather than posterior or covariance draws.

Every population and ensemble specification has a canonical SHA-256 identity.
The pseudo-random mapping is specified in this module and depends only on the
locked seed, the bound population digest, and immutable member identifiers.
It does not use NumPy's process-global random state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping


SCIENCE_INPUT_SCHEMA = "jx.long-horizon.science-inputs.v1"
POPULATION_SCHEMA = "jx.long-horizon.orbital-population.v1"
PERTURBATION_SCHEMA = "jx.long-horizon.perturbation-ensemble.v1"
MATERIALIZED_ENSEMBLE_SCHEMA = (
    "jx.long-horizon.materialized-perturbation-ensemble.v1"
)
PERTURBATION_GENERATOR_ID = "SHA256_CENTERED_OPEN_UNIFORM_V1"
SCIENCE_INPUT_CLAIM_CEILING = (
    "OBSERVED_OBJECT_BENCHMARK_AND_NUMERICAL_SENSITIVITY_ONLY"
)
SCIENCE_INPUT_NONCLAIMS = (
    "not an observationally complete or debiased population model",
    "not a posterior or full-covariance uncertainty sample",
    "not evidence for detection, exclusion, orbit, mass, or sky position",
    "not a scientific result without separately qualified dynamics and inference",
)
NON_INFERENCE_STATUS = MappingProxyType(
    {
        "observationally_complete": False,
        "observationally_fitted": False,
        "uncertainty_calibrated": False,
        "detection_claim": False,
        "exclusion_claim": False,
        "science_result": False,
    }
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ELEMENT_FIELDS = (
    "semimajor_axis_AU",
    "eccentricity",
    "inclination_rad",
    "longitude_of_ascending_node_rad",
    "argument_of_periapsis_rad",
    "mean_anomaly_rad",
)
_ANGULAR_FIELDS = frozenset(_ELEMENT_FIELDS[2:])


class ScienceInputContractError(ValueError):
    """A long-horizon science-input declaration is incomplete or ambiguous."""


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ScienceInputContractError(
            f"{label} must match {_IDENTIFIER.pattern!r}"
        )
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ScienceInputContractError(
            f"{label} must be a nonempty, trimmed string"
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ScienceInputContractError(
            f"{label} must be a finite built-in float"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScienceInputContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _canonical_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def _float_record(value: float) -> str:
    return value.hex()


@dataclass(frozen=True, slots=True)
class ObservedOrbitSource:
    """Exact provenance and one-sigma element errors for one JPL SBDB row."""

    spk_id: int
    primary_designation: str
    full_name: str
    orbit_solution_id: str
    source_code: str
    data_arc_days: int
    observations_used: int
    sigma_semimajor_axis_AU: float
    sigma_eccentricity: float
    sigma_inclination_rad: float
    sigma_longitude_of_ascending_node_rad: float
    sigma_argument_of_periapsis_rad: float
    sigma_mean_anomaly_rad: float
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.spk_id) is not int or self.spk_id <= 0:
            raise ScienceInputContractError("spk_id must be a positive integer")
        for value, label in (
            (self.primary_designation, "primary_designation"),
            (self.full_name, "full_name"),
            (self.orbit_solution_id, "orbit_solution_id"),
            (self.source_code, "source_code"),
        ):
            _text(value, label)
        for value, label in (
            (self.data_arc_days, "data_arc_days"),
            (self.observations_used, "observations_used"),
        ):
            if type(value) is not int or value <= 0:
                raise ScienceInputContractError(f"{label} must be positive")
        for name in (
            "sigma_semimajor_axis_AU",
            "sigma_eccentricity",
            "sigma_inclination_rad",
            "sigma_longitude_of_ascending_node_rad",
            "sigma_argument_of_periapsis_rad",
            "sigma_mean_anomaly_rad",
        ):
            if _finite_float(getattr(self, name), name) <= 0.0:
                raise ScienceInputContractError(f"{name} must be positive")
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.observed-orbit-source.v1",
                self._canonical_record(),
            ),
        )

    def sigma_for(self, element_name: str) -> float:
        mapping = {
            "semimajor_axis_AU": self.sigma_semimajor_axis_AU,
            "eccentricity": self.sigma_eccentricity,
            "inclination_rad": self.sigma_inclination_rad,
            "longitude_of_ascending_node_rad": (
                self.sigma_longitude_of_ascending_node_rad
            ),
            "argument_of_periapsis_rad": self.sigma_argument_of_periapsis_rad,
            "mean_anomaly_rad": self.sigma_mean_anomaly_rad,
        }
        try:
            return mapping[element_name]
        except KeyError as exc:
            raise ScienceInputContractError(
                f"unknown source-uncertainty element {element_name!r}"
            ) from exc

    def _canonical_record(self) -> dict[str, Any]:
        return {
            "spk_id": self.spk_id,
            "primary_designation": self.primary_designation,
            "full_name": self.full_name,
            "orbit_solution_id": self.orbit_solution_id,
            "source_code": self.source_code,
            "data_arc_days": self.data_arc_days,
            "observations_used": self.observations_used,
            "sigmas": {
                name: _float_record(getattr(self, name))
                for name in (
                    "sigma_semimajor_axis_AU",
                    "sigma_eccentricity",
                    "sigma_inclination_rad",
                    "sigma_longitude_of_ascending_node_rad",
                    "sigma_argument_of_periapsis_rad",
                    "sigma_mean_anomaly_rad",
                )
            },
        }


@dataclass(frozen=True, slots=True)
class OrbitalElements:
    """One explicit heliocentric osculating-element regime probe."""

    object_id: str
    semimajor_axis_AU: float
    eccentricity: float
    inclination_rad: float
    longitude_of_ascending_node_rad: float
    argument_of_periapsis_rad: float
    mean_anomaly_rad: float
    observed_source: ObservedOrbitSource | None = None

    def __post_init__(self) -> None:
        _identifier(self.object_id, "object_id")
        semimajor_axis = _finite_float(
            self.semimajor_axis_AU, "semimajor_axis_AU"
        )
        eccentricity = _finite_float(self.eccentricity, "eccentricity")
        if semimajor_axis <= 0.0 or not 0.0 <= eccentricity < 1.0:
            raise ScienceInputContractError(
                "bound screening elements require a > 0 and 0 <= e < 1"
            )
        if semimajor_axis * (1.0 - eccentricity) <= 0.0:
            raise ScienceInputContractError("perihelion distance must be positive")
        inclination = _finite_float(self.inclination_rad, "inclination_rad")
        if not 0.0 <= inclination <= math.pi:
            raise ScienceInputContractError("inclination_rad must be in [0, pi]")
        for label in _ELEMENT_FIELDS[3:]:
            angle = _finite_float(getattr(self, label), label)
            if not 0.0 <= angle < math.tau:
                raise ScienceInputContractError(f"{label} must be in [0, 2*pi)")
        if self.observed_source is not None and type(
            self.observed_source
        ) is not ObservedOrbitSource:
            raise ScienceInputContractError(
                "observed_source must be an exact ObservedOrbitSource"
            )

    @property
    def perihelion_AU(self) -> float:
        return self.semimajor_axis_AU * (1.0 - self.eccentricity)

    def _canonical_record(self) -> dict[str, str]:
        return {
            "object_id": self.object_id,
            **{
                name: _float_record(getattr(self, name))
                for name in _ELEMENT_FIELDS
            },
            "observed_source": (
                None
                if self.observed_source is None
                else {
                    **self.observed_source._canonical_record(),
                    "content_sha256": self.observed_source.content_sha256,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class ScientificPopulationBlock:
    """An explicit orbital-regime block locked by canonical content hash."""

    population_id: str
    title: str
    construction: str
    epoch_jd_tdb: float
    frame: str
    members: tuple[OrbitalElements, ...]
    source_authority: str
    source_schema_version: str
    source_query_uri: str
    source_artifact_path: str
    source_artifact_sha256: str
    selection_rule: str
    acquisition_date: str
    claim_ceiling: str = SCIENCE_INPUT_CLAIM_CEILING
    nonclaims: tuple[str, ...] = SCIENCE_INPUT_NONCLAIMS
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.population_id, "population_id")
        _text(self.title, "title")
        _text(self.construction, "construction")
        epoch = _finite_float(self.epoch_jd_tdb, "epoch_jd_tdb")
        if epoch <= 0.0:
            raise ScienceInputContractError("epoch_jd_tdb must be positive")
        _text(self.frame, "frame")
        for value, label in (
            (self.source_authority, "source_authority"),
            (self.source_schema_version, "source_schema_version"),
            (self.source_query_uri, "source_query_uri"),
            (self.source_artifact_path, "source_artifact_path"),
            (self.selection_rule, "selection_rule"),
            (self.acquisition_date, "acquisition_date"),
        ):
            _text(value, label)
        _sha256(self.source_artifact_sha256, "source_artifact_sha256")
        if type(self.members) is not tuple or not self.members:
            raise ScienceInputContractError("members must be a nonempty exact tuple")
        if any(type(member) is not OrbitalElements for member in self.members):
            raise ScienceInputContractError(
                "members must contain exact OrbitalElements values"
            )
        if any(member.observed_source is None for member in self.members):
            raise ScienceInputContractError(
                "scientific population members require observed-source provenance"
            )
        member_ids = tuple(member.object_id for member in self.members)
        if len(set(member_ids)) != len(member_ids):
            raise ScienceInputContractError("population object IDs must be unique")
        if (
            self.claim_ceiling != SCIENCE_INPUT_CLAIM_CEILING
            or self.nonclaims != SCIENCE_INPUT_NONCLAIMS
        ):
            raise ScienceInputContractError(
                "science-input claim ceiling cannot be widened"
            )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.orbital-population.v1",
                self._canonical_record(),
            ),
        )

    def _canonical_record(self) -> dict[str, Any]:
        return {
            "schema": POPULATION_SCHEMA,
            "population_id": self.population_id,
            "title": self.title,
            "construction": self.construction,
            "epoch_jd_tdb_hex": _float_record(self.epoch_jd_tdb),
            "frame": self.frame,
            "source_authority": self.source_authority,
            "source_schema_version": self.source_schema_version,
            "source_query_uri": self.source_query_uri,
            "source_artifact_path": self.source_artifact_path,
            "source_artifact_sha256": self.source_artifact_sha256,
            "selection_rule": self.selection_rule,
            "acquisition_date": self.acquisition_date,
            "members": [member._canonical_record() for member in self.members],
            "claim_ceiling": self.claim_ceiling,
            "nonclaims": list(self.nonclaims),
        }


@dataclass(frozen=True, slots=True)
class PerturbationAxis:
    """One bounded independent sensitivity offset, never a covariance axis."""

    element_name: str
    half_width: float
    unit: str
    width_mode: str = "SOURCE_SIGMA_MULTIPLIER"
    distribution: str = "CENTERED_OPEN_UNIFORM"

    def __post_init__(self) -> None:
        if self.element_name not in _ELEMENT_FIELDS:
            raise ScienceInputContractError(
                f"element_name must be one of {_ELEMENT_FIELDS}"
            )
        width = _finite_float(self.half_width, "half_width")
        if width <= 0.0:
            raise ScienceInputContractError("half_width must be positive")
        _text(self.unit, "unit")
        if self.width_mode != "SOURCE_SIGMA_MULTIPLIER":
            raise ScienceInputContractError(
                "perturbation widths must scale source-reported one-sigma errors"
            )
        if self.distribution != "CENTERED_OPEN_UNIFORM":
            raise ScienceInputContractError(
                "only the locked centered-open-uniform distribution is supported"
            )

    def _canonical_record(self) -> dict[str, str]:
        return {
            "element_name": self.element_name,
            "half_width_hex": _float_record(self.half_width),
            "unit": self.unit,
            "width_mode": self.width_mode,
            "distribution": self.distribution,
        }


@dataclass(frozen=True, slots=True)
class SeededPerturbationEnsembleSpec:
    """A deterministic sensitivity ensemble bound to one population digest."""

    ensemble_id: str
    population_id: str
    population_sha256: str
    seed: int
    realizations_per_object: int
    axes: tuple[PerturbationAxis, ...]
    interpretation: str
    generator_id: str = PERTURBATION_GENERATOR_ID
    claim_ceiling: str = SCIENCE_INPUT_CLAIM_CEILING
    nonclaims: tuple[str, ...] = SCIENCE_INPUT_NONCLAIMS
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.ensemble_id, "ensemble_id")
        _identifier(self.population_id, "population_id")
        _sha256(self.population_sha256, "population_sha256")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ScienceInputContractError("seed must be a uint64 integer")
        if (
            type(self.realizations_per_object) is not int
            or self.realizations_per_object <= 0
        ):
            raise ScienceInputContractError(
                "realizations_per_object must be a positive integer"
            )
        if type(self.axes) is not tuple or not self.axes:
            raise ScienceInputContractError("axes must be a nonempty exact tuple")
        if any(type(axis) is not PerturbationAxis for axis in self.axes):
            raise ScienceInputContractError(
                "axes must contain exact PerturbationAxis values"
            )
        axis_names = tuple(axis.element_name for axis in self.axes)
        if len(set(axis_names)) != len(axis_names):
            raise ScienceInputContractError("perturbation axes must be unique")
        _text(self.interpretation, "interpretation")
        if "not covariance" not in self.interpretation.lower():
            raise ScienceInputContractError(
                "interpretation must state that offsets are not covariance draws"
            )
        if (
            self.generator_id != PERTURBATION_GENERATOR_ID
            or self.claim_ceiling != SCIENCE_INPUT_CLAIM_CEILING
            or self.nonclaims != SCIENCE_INPUT_NONCLAIMS
        ):
            raise ScienceInputContractError(
                "ensemble generator or claim ceiling cannot be widened"
            )
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.perturbation-ensemble.v1",
                self._canonical_record(),
            ),
        )

    def _canonical_record(self) -> dict[str, Any]:
        return {
            "schema": PERTURBATION_SCHEMA,
            "ensemble_id": self.ensemble_id,
            "population_id": self.population_id,
            "population_sha256": self.population_sha256,
            "seed": self.seed,
            "realizations_per_object": self.realizations_per_object,
            "axes": [axis._canonical_record() for axis in self.axes],
            "interpretation": self.interpretation,
            "generator_id": self.generator_id,
            "claim_ceiling": self.claim_ceiling,
            "nonclaims": list(self.nonclaims),
        }


@dataclass(frozen=True, slots=True)
class PerturbedOrbit:
    """One deterministic sensitivity realization and its exact offsets."""

    parent_object_id: str
    realization_index: int
    elements: OrbitalElements
    offsets: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _identifier(self.parent_object_id, "parent_object_id")
        if type(self.realization_index) is not int or self.realization_index < 0:
            raise ScienceInputContractError(
                "realization_index must be a nonnegative integer"
            )
        if type(self.elements) is not OrbitalElements:
            raise ScienceInputContractError(
                "elements must be an exact OrbitalElements value"
            )
        if type(self.offsets) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or item[0] not in _ELEMENT_FIELDS
            or type(item[1]) is not float
            or not math.isfinite(item[1])
            for item in self.offsets
        ):
            raise ScienceInputContractError("offsets must be immutable finite pairs")

    def _canonical_record(self) -> dict[str, Any]:
        return {
            "parent_object_id": self.parent_object_id,
            "realization_index": self.realization_index,
            "elements": self.elements._canonical_record(),
            "offsets": [
                {"element_name": name, "offset_hex": _float_record(value)}
                for name, value in self.offsets
            ],
        }


@dataclass(frozen=True, slots=True)
class MaterializedPerturbationEnsemble:
    """Immutable deterministic expansion of one locked ensemble spec."""

    ensemble_id: str
    ensemble_sha256: str
    population_sha256: str
    members: tuple[PerturbedOrbit, ...]
    claim_ceiling: str = SCIENCE_INPUT_CLAIM_CEILING
    nonclaims: tuple[str, ...] = SCIENCE_INPUT_NONCLAIMS
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.ensemble_id, "ensemble_id")
        _sha256(self.ensemble_sha256, "ensemble_sha256")
        _sha256(self.population_sha256, "population_sha256")
        if type(self.members) is not tuple or not self.members or any(
            type(member) is not PerturbedOrbit for member in self.members
        ):
            raise ScienceInputContractError(
                "members must be a nonempty exact PerturbedOrbit tuple"
            )
        if (
            self.claim_ceiling != SCIENCE_INPUT_CLAIM_CEILING
            or self.nonclaims != SCIENCE_INPUT_NONCLAIMS
        ):
            raise ScienceInputContractError(
                "materialized ensemble claim ceiling cannot be widened"
            )
        record = {
            "schema": MATERIALIZED_ENSEMBLE_SCHEMA,
            "ensemble_id": self.ensemble_id,
            "ensemble_sha256": self.ensemble_sha256,
            "population_sha256": self.population_sha256,
            "members": [member._canonical_record() for member in self.members],
            "claim_ceiling": self.claim_ceiling,
            "nonclaims": list(self.nonclaims),
        }
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                b"jx.long-horizon.materialized-perturbation-ensemble.v1",
                record,
            ),
        )


def _open_uniform(
    spec: SeededPerturbationEnsembleSpec,
    population: ScientificPopulationBlock,
    object_id: str,
    realization_index: int,
    element_name: str,
) -> float:
    digest = hashlib.sha256()
    digest.update(b"jx.long-horizon.perturbation-draw.v1\0")
    digest.update(spec.seed.to_bytes(8, "big", signed=False))
    digest.update(bytes.fromhex(spec.content_sha256))
    digest.update(bytes.fromhex(population.content_sha256))
    for value in (object_id, str(realization_index), element_name):
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    integer = int.from_bytes(digest.digest()[:8], "big", signed=False)
    return math.ldexp(integer + 0.5, -64)


def materialize_perturbation_ensemble(
    spec: SeededPerturbationEnsembleSpec,
    population: ScientificPopulationBlock,
) -> MaterializedPerturbationEnsemble:
    """Expand a locked sensitivity ensemble without global RNG state."""

    if type(spec) is not SeededPerturbationEnsembleSpec:
        raise ScienceInputContractError(
            "spec must be an exact SeededPerturbationEnsembleSpec"
        )
    if type(population) is not ScientificPopulationBlock:
        raise ScienceInputContractError(
            "population must be an exact ScientificPopulationBlock"
        )
    if (
        spec.population_id != population.population_id
        or spec.population_sha256 != population.content_sha256
    ):
        raise ScienceInputContractError(
            "ensemble is not bound to the supplied population content"
        )

    members: list[PerturbedOrbit] = []
    for parent in population.members:
        for realization_index in range(spec.realizations_per_object):
            values = {name: getattr(parent, name) for name in _ELEMENT_FIELDS}
            offsets: list[tuple[str, float]] = []
            for axis in spec.axes:
                unit = _open_uniform(
                    spec,
                    population,
                    parent.object_id,
                    realization_index,
                    axis.element_name,
                )
                if parent.observed_source is None:
                    raise ScienceInputContractError(
                        "source-sigma perturbations require observed provenance"
                    )
                half_width = (
                    axis.half_width
                    * parent.observed_source.sigma_for(axis.element_name)
                )
                offset = float((2.0 * unit - 1.0) * half_width)
                values[axis.element_name] += offset
                if axis.element_name in _ANGULAR_FIELDS:
                    values[axis.element_name] %= math.tau
                offsets.append((axis.element_name, offset))
            element_id = (
                f"{parent.object_id}.p{realization_index:04d}"
            )
            elements = OrbitalElements(object_id=element_id, **values)
            members.append(
                PerturbedOrbit(
                    parent_object_id=parent.object_id,
                    realization_index=realization_index,
                    elements=elements,
                    offsets=tuple(offsets),
                )
            )
    return MaterializedPerturbationEnsemble(
        ensemble_id=spec.ensemble_id,
        ensemble_sha256=spec.content_sha256,
        population_sha256=population.content_sha256,
        members=tuple(members),
    )


_FRAME = "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_OSCULATING"
_EPOCH_JD_TDB = 2_461_200.5
_SOURCE_AUTHORITY = "NASA/JPL SBDB (Small-Body DataBase) Query API"
_SOURCE_SCHEMA_VERSION = "1.0"
_ACQUISITION_DATE = "2026-10-05"
_DETACHED_SOURCE_PATH = (
    "jxplanetx/data/long_horizon_science_inputs_v1/raw/"
    "jpl_sbdb_detached_query_20261005.json"
)
_DETACHED_SOURCE_SHA256 = (
    "91647cabbce60eb362b54ddf5be525188e5c58c8db1ade66c0bc6398be78df32"
)
_DETACHED_QUERY_URI = (
    "https://ssd-api.jpl.nasa.gov/sbdb_query.api?"
    "fields=spkid,pdes,full_name,orbit_id,epoch,equinox,e,a,q,i,om,w,ma,"
    "sigma_e,sigma_a,sigma_i,sigma_om,sigma_w,sigma_ma,condition_code,"
    "data_arc,n_obs_used,source&sb-class=TNO&sb-cdata=%7B%22AND%22%3A%5B%22"
    "q%7CGE%7C40%22%2C%22a%7CGE%7C150%22%2C%22e%7CLT%7C1%22%2C%22"
    "condition_code%7CLE%7C5%22%5D%7D&sort=q&limit=50&full-prec=true"
)
_CONTROL_SOURCE_PATH = (
    "jxplanetx/data/long_horizon_science_inputs_v1/raw/"
    "jpl_sbdb_scattering_control_query_20261005.json"
)
_CONTROL_SOURCE_SHA256 = (
    "3d82cc80b73f9fcb5e7969375129524ba0356926c7a14ac82955d0083176a224"
)
_CONTROL_QUERY_URI = (
    "https://ssd-api.jpl.nasa.gov/sbdb_query.api?"
    "fields=spkid,pdes,name,full_name,orbit_id,epoch,equinox,e,a,q,i,om,w,ma,"
    "sigma_e,sigma_a,sigma_i,sigma_om,sigma_w,sigma_ma,condition_code,"
    "data_arc,n_obs_used,source&sb-class=TNO&sb-cdata=%7B%22AND%22%3A%5B%22"
    "q%7CGE%7C30%22%2C%22q%7CLT%7C40%22%2C%22a%7CGE%7C50%22%2C%22"
    "a%7CLT%7C150%22%2C%22e%7CLT%7C1%22%2C%22condition_code%7CLE%7C3%22"
    "%5D%7D&sort=a&full-prec=true"
)


def _observed_source(
    spk_id: int,
    primary_designation: str,
    full_name: str,
    orbit_solution_id: str,
    data_arc_days: int,
    observations_used: int,
    sigma_a_AU: float,
    sigma_e: float,
    sigma_i_deg: float,
    sigma_node_deg: float,
    sigma_peri_deg: float,
    sigma_mean_anomaly_deg: float,
) -> ObservedOrbitSource:
    return ObservedOrbitSource(
        spk_id=spk_id,
        primary_designation=primary_designation,
        full_name=full_name,
        orbit_solution_id=orbit_solution_id,
        source_code="ORB",
        data_arc_days=data_arc_days,
        observations_used=observations_used,
        sigma_semimajor_axis_AU=sigma_a_AU,
        sigma_eccentricity=sigma_e,
        sigma_inclination_rad=math.radians(sigma_i_deg),
        sigma_longitude_of_ascending_node_rad=math.radians(sigma_node_deg),
        sigma_argument_of_periapsis_rad=math.radians(sigma_peri_deg),
        sigma_mean_anomaly_rad=math.radians(sigma_mean_anomaly_deg),
    )


DETACHED_TNO_SCREEN_V1 = ScientificPopulationBlock(
    population_id="jpl-observed-detached-tno-v1",
    title="JPL-observed detached trans-Neptunian benchmark",
    construction=(
        "Six NASA/JPL SBDB rows selected without dynamical outcomes: require "
        "complete source uncertainties and ORB provenance, then take the six "
        "longest observational arcs (ties by observation count then SPK-ID)."
    ),
    epoch_jd_tdb=_EPOCH_JD_TDB,
    frame=_FRAME,
    source_authority=_SOURCE_AUTHORITY,
    source_schema_version=_SOURCE_SCHEMA_VERSION,
    source_query_uri=_DETACHED_QUERY_URI,
    source_artifact_path=_DETACHED_SOURCE_PATH,
    source_artifact_sha256=_DETACHED_SOURCE_SHA256,
    selection_rule=(
        "source=ORB; all six element sigmas defined; sort data_arc_days "
        "descending, observations_used descending, spk_id ascending; take 6"
    ),
    acquisition_date=_ACQUISITION_DATE,
    members=(
        OrbitalElements("jpl-20090377", 543.7195289104732, 0.8598824585187618, math.radians(11.92527582847476), math.radians(144.5061662673739), math.radians(311.0987725939751), math.radians(358.5956944005428), observed_source=_observed_source(20090377, "90377", "90377 Sedna (2003 VB12)", "JPL 51", 12939, 671, 0.1459, 3.7251e-5, 3.3169e-6, 0.00026058, 0.0019143, 0.0005731)),
        OrbitalElements("jpl-20474640", 342.1499424203475, 0.861547579461181, math.radians(25.51048085878516), math.radians(66.03582012284039), math.radians(327.2807650284388), math.radians(0.9438834766288601), observed_source=_observed_source(20474640, "474640", "474640 Alicanto (2004 VN112)", "12", 8737, 44, 1.0146, 0.00040111, 0.00019023, 0.00026204, 0.0028648, 0.0042404)),
        OrbitalElements("jpl-54526835", 247.4285195857868, 0.7335661561347354, math.radians(11.00095782821624), math.radians(72.07902619292427), math.radians(198.7037510664639), math.radians(356.5705110200873), observed_source=_observed_source(54526835, "2023 KQ14", "(2023 KQ14)", "3", 7323, 88, 0.28995, 0.00031193, 0.00070253, 0.0010078, 0.018953, 0.0061541)),
        OrbitalElements("jpl-20541132", 1345.9746784971, 0.9519448010157889, math.radians(11.67889561905244), math.radians(301.1346719755806), math.radians(118.2317322688105), math.radians(359.6190139698275), observed_source=_observed_source(20541132, "541132", "541132 Leleakuhonua (2015 TG387)", "7", 7022, 44, 10.094, 0.00035882, 0.00016023, 0.0018761, 0.009741, 0.0043212)),
        OrbitalElements("jpl-50666516", 267.5209861941053, 0.6985589359876868, math.radians(24.00179208811373), math.radians(90.8926204194851), math.radians(294.2852296107872), math.radians(3.834281215098069), observed_source=_observed_source(50666516, "2012 VP113", "(2012 VP113)", "11", 6187, 55, 0.27641, 0.00029374, 0.00024748, 0.0005882, 0.018843, 0.0068036)),
        OrbitalElements("jpl-20148209", 229.2563139415033, 0.8073365317057665, math.radians(22.70799015441454), math.radians(128.2037509183512), math.radians(317.0408950104643), math.radians(6.318034542288514), observed_source=_observed_source(20148209, "148209", "148209 (2000 CR105)", "5", 5547, 57, 0.68675, 0.00057361, 0.00058393, 0.00029856, 0.011587, 0.02909)),
    ),
)


SCATTERING_DISK_CONTROL_V1 = ScientificPopulationBlock(
    population_id="jpl-observed-scattering-control-v1",
    title="JPL-observed scattering-disk benchmark control",
    construction=(
        "Six unnamed NASA/JPL SBDB rows selected without dynamical outcomes: "
        "require complete source uncertainties and ORB provenance, then take "
        "the six longest observational arcs (ties by observations then SPK-ID)."
    ),
    epoch_jd_tdb=_EPOCH_JD_TDB,
    frame=_FRAME,
    source_authority=_SOURCE_AUTHORITY,
    source_schema_version=_SOURCE_SCHEMA_VERSION,
    source_query_uri=_CONTROL_QUERY_URI,
    source_artifact_path=_CONTROL_SOURCE_PATH,
    source_artifact_sha256=_CONTROL_SOURCE_SHA256,
    selection_rule=(
        "name=null; source=ORB; all six element sigmas defined; sort "
        "data_arc_days descending, observations_used descending, spk_id "
        "ascending; take 6"
    ),
    acquisition_date=_ACQUISITION_DATE,
    members=(
        OrbitalElements("jpl-20574372", 78.09363316381607, 0.4988701464793244, math.radians(32.06386260847041), math.radians(147.0462304885865), math.radians(9.496145292923686), math.radians(39.00158077771329), observed_source=_observed_source(20574372, "574372", "574372 (2010 JO179)", "14", 27175, 218, 0.0036096, 1.7544e-5, 9.26e-6, 0.00010135, 0.0019924, 0.003269)),
        OrbitalElements("jpl-20303775", 111.7272902357856, 0.6686928465820892, math.radians(14.01023423630377), math.radians(78.51532092257791), math.radians(224.1252348407445), math.radians(16.74680404631517), observed_source=_observed_source(20303775, "303775", "303775 (2005 QU182)", "29", 18818, 626, 0.0054239, 1.3695e-5, 1.6148e-5, 0.00011291, 0.00097314, 0.0013345)),
        OrbitalElements("jpl-20026181", 92.78882564294666, 0.5888130202724389, math.radians(13.3537460632102), math.radians(194.18417395), math.radians(356.1020162135479), math.radians(14.82871470714553), observed_source=_observed_source(20026181, "26181", "26181 (1996 GQ21)", "25", 16564, 459, 0.0037731, 1.4798e-5, 1.8464e-5, 8.2012e-5, 0.00067755, 0.0010335)),
        OrbitalElements("jpl-20042301", 51.68239129795629, 0.2805075574056742, math.radians(0.7510091755756962), math.radians(302.4037703262449), math.radians(343.1295377754446), math.radians(86.69212704756386), observed_source=_observed_source(20042301, "42301", "42301 (2001 UR163)", "28", 15872, 443, 0.00094131, 6.046e-6, 4.2761e-6, 0.0031634, 0.0041592, 0.0032556)),
        OrbitalElements("jpl-20026375", 55.61391834367028, 0.4206896190844759, math.radians(7.610567083727258), math.radians(322.8597924267265), math.radians(159.5941686692768), math.radians(34.89083332820755), observed_source=_observed_source(20026375, "26375", "26375 (1999 DE9)", "JPL 38", 13175, 553, 0.0011385, 8.7573e-6, 1.5479e-5, 0.00011258, 0.0010232, 0.0014088)),
        OrbitalElements("jpl-20015874", 84.7160141732945, 0.5856282476985989, math.radians(23.96488441448197), math.radians(217.7307666730197), math.radians(185.1373827636416), math.radians(11.4275870702862), observed_source=_observed_source(20015874, "15874", "15874 (1996 TL66)", "JPL 34", 10699, 586, 0.0027281, 1.2165e-5, 1.272e-5, 2.5628e-5, 0.00032816, 0.00061897)),
    ),
)


PREDECLARED_POPULATIONS = MappingProxyType(
    {
        DETACHED_TNO_SCREEN_V1.population_id: DETACHED_TNO_SCREEN_V1,
        SCATTERING_DISK_CONTROL_V1.population_id: SCATTERING_DISK_CONTROL_V1,
    }
)


def _sensitivity_axes() -> tuple[PerturbationAxis, ...]:
    return (
        PerturbationAxis("semimajor_axis_AU", 3.0, "source_sigma"),
        PerturbationAxis("eccentricity", 3.0, "source_sigma"),
        PerturbationAxis("inclination_rad", 3.0, "source_sigma"),
        PerturbationAxis(
            "longitude_of_ascending_node_rad", 3.0, "source_sigma"
        ),
        PerturbationAxis(
            "argument_of_periapsis_rad", 3.0, "source_sigma"
        ),
        PerturbationAxis("mean_anomaly_rad", 3.0, "source_sigma"),
    )


DETACHED_TNO_SENSITIVITY_V1 = SeededPerturbationEnsembleSpec(
    ensemble_id="detached-tno-sensitivity-v1",
    population_id=DETACHED_TNO_SCREEN_V1.population_id,
    population_sha256=DETACHED_TNO_SCREEN_V1.content_sha256,
    seed=0x4A58504C414E4554,
    realizations_per_object=8,
    axes=_sensitivity_axes(),
    interpretation=(
        "Independent bounded offsets within three source-reported standard "
        "errors per element; explicitly not covariance draws or posterior draws."
    ),
)


SCATTERING_DISK_SENSITIVITY_V1 = SeededPerturbationEnsembleSpec(
    ensemble_id="scattering-disk-sensitivity-v1",
    population_id=SCATTERING_DISK_CONTROL_V1.population_id,
    population_sha256=SCATTERING_DISK_CONTROL_V1.content_sha256,
    seed=0x5245535452494354,
    realizations_per_object=8,
    axes=_sensitivity_axes(),
    interpretation=(
        "Independent bounded offsets within three source-reported standard "
        "errors per element; explicitly not covariance draws or posterior draws."
    ),
)


PREDECLARED_PERTURBATION_ENSEMBLES = MappingProxyType(
    {
        DETACHED_TNO_SENSITIVITY_V1.ensemble_id: DETACHED_TNO_SENSITIVITY_V1,
        SCATTERING_DISK_SENSITIVITY_V1.ensemble_id: (
            SCATTERING_DISK_SENSITIVITY_V1
        ),
    }
)


SCIENCE_INPUTS_CONTENT_SHA256 = _canonical_sha256(
    b"jx.long-horizon.science-input-inventory.v1",
    {
        "schema": SCIENCE_INPUT_SCHEMA,
        "populations": [
            {
                "population_id": item.population_id,
                "content_sha256": item.content_sha256,
            }
            for item in PREDECLARED_POPULATIONS.values()
        ],
        "perturbation_ensembles": [
            {
                "ensemble_id": item.ensemble_id,
                "content_sha256": item.content_sha256,
            }
            for item in PREDECLARED_PERTURBATION_ENSEMBLES.values()
        ],
        "claim_ceiling": SCIENCE_INPUT_CLAIM_CEILING,
        "nonclaims": list(SCIENCE_INPUT_NONCLAIMS),
    },
)


def get_population(population_id: str) -> ScientificPopulationBlock:
    """Return one predeclared population; unknown identities fail closed."""

    _identifier(population_id, "population_id")
    try:
        return PREDECLARED_POPULATIONS[population_id]
    except KeyError as exc:
        raise ScienceInputContractError(
            f"unknown predeclared population {population_id!r}"
        ) from exc


def get_perturbation_ensemble(
    ensemble_id: str,
) -> SeededPerturbationEnsembleSpec:
    """Return one predeclared perturbation spec; unknown IDs fail closed."""

    _identifier(ensemble_id, "ensemble_id")
    try:
        return PREDECLARED_PERTURBATION_ENSEMBLES[ensemble_id]
    except KeyError as exc:
        raise ScienceInputContractError(
            f"unknown predeclared perturbation ensemble {ensemble_id!r}"
        ) from exc


__all__ = [
    "DETACHED_TNO_SCREEN_V1",
    "DETACHED_TNO_SENSITIVITY_V1",
    "MATERIALIZED_ENSEMBLE_SCHEMA",
    "NON_INFERENCE_STATUS",
    "ObservedOrbitSource",
    "OrbitalElements",
    "PERTURBATION_GENERATOR_ID",
    "PERTURBATION_SCHEMA",
    "POPULATION_SCHEMA",
    "PREDECLARED_PERTURBATION_ENSEMBLES",
    "PREDECLARED_POPULATIONS",
    "PerturbationAxis",
    "PerturbedOrbit",
    "SCATTERING_DISK_CONTROL_V1",
    "SCATTERING_DISK_SENSITIVITY_V1",
    "SCIENCE_INPUT_CLAIM_CEILING",
    "SCIENCE_INPUT_NONCLAIMS",
    "SCIENCE_INPUT_SCHEMA",
    "SCIENCE_INPUTS_CONTENT_SHA256",
    "ScienceInputContractError",
    "ScientificPopulationBlock",
    "SeededPerturbationEnsembleSpec",
    "MaterializedPerturbationEnsemble",
    "get_perturbation_ensemble",
    "get_population",
    "materialize_perturbation_ensemble",
]
