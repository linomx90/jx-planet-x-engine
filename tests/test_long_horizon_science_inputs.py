from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import unittest

from jxplanetx.cuda_mechanism_schedules import DE441_CONTROL_EPOCH_JD_TDB
from jxplanetx.long_horizon import (
    DETACHED_TNO_SCREEN_V1,
    DETACHED_TNO_SENSITIVITY_V1,
    NON_INFERENCE_STATUS,
    SCATTERING_DISK_CONTROL_V1,
    SCATTERING_DISK_SENSITIVITY_V1,
    SCIENCE_INPUTS_CONTENT_SHA256,
    ScienceInputContractError,
    get_population,
    materialize_perturbation_ensemble,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "long_horizon_science_inputs_v1"
SIGMA_FIELDS = ("sigma_a", "sigma_e", "sigma_i", "sigma_om", "sigma_w", "sigma_ma")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["signature"] != {
        "source": "NASA/JPL SBDB (Small-Body DataBase) Query API",
        "version": "1.0",
    }:
        raise AssertionError("unexpected SBDB signature")
    fields = document["fields"]
    rows = [dict(zip(fields, values, strict=True)) for values in document["data"]]
    if int(document["count"]) != len(rows):
        raise AssertionError("SBDB count mismatch")
    return rows


def _selected(rows: list[dict[str, object]], *, unnamed: bool) -> list[dict[str, object]]:
    eligible = []
    for row in rows:
        if unnamed and row["name"] is not None:
            continue
        if row["source"] != "ORB" or row["data_arc"] is None:
            continue
        if any(row[field] is None for field in SIGMA_FIELDS):
            continue
        eligible.append(row)
    eligible.sort(
        key=lambda row: (
            -int(row["data_arc"]),
            -int(row["n_obs_used"]),
            int(row["spkid"]),
        )
    )
    return eligible[:6]


class LongHorizonScienceInputsTests(unittest.TestCase):
    def test_retained_raw_artifacts_and_manifest_are_exact(self) -> None:
        manifest = json.loads((RUN / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["epoch_jd_tdb"], DE441_CONTROL_EPOCH_JD_TDB)
        for record in manifest["sources"].values():
            path = RUN / record["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(_sha256(path), record["sha256"])
            self.assertEqual(len(_rows(path)), record["rows"])
        for line in (RUN / "SHA256SUMS").read_text(encoding="ascii").splitlines():
            digest, relative = line.split("  ", 1)
            self.assertEqual(_sha256(RUN / relative), digest)

    def test_installed_package_retains_the_authoritative_raw_inputs(self) -> None:
        packaged = resources.files("jxplanetx").joinpath(
            "data", "long_horizon_science_inputs_v1"
        )
        for relative in (
            "MANIFEST.json",
            "README.md",
            "SHA256SUMS",
            "raw/jpl_sbdb_detached_query_20261005.json",
            "raw/jpl_sbdb_scattering_control_query_20261005.json",
        ):
            package_bytes = packaged.joinpath(*relative.split("/")).read_bytes()
            self.assertEqual(package_bytes, (RUN / relative).read_bytes())
        for population in (DETACHED_TNO_SCREEN_V1, SCATTERING_DISK_CONTROL_V1):
            self.assertTrue(
                population.source_artifact_path.startswith(
                    "jxplanetx/data/long_horizon_science_inputs_v1/raw/"
                )
            )

    def test_nominal_elements_are_exact_selected_jpl_rows_at_de441_epoch(self) -> None:
        cases = (
            (
                DETACHED_TNO_SCREEN_V1,
                RUN / "raw" / "jpl_sbdb_detached_query_20261005.json",
                False,
            ),
            (
                SCATTERING_DISK_CONTROL_V1,
                RUN / "raw" / "jpl_sbdb_scattering_control_query_20261005.json",
                True,
            ),
        )
        for population, path, unnamed in cases:
            self.assertEqual(population.epoch_jd_tdb, DE441_CONTROL_EPOCH_JD_TDB)
            self.assertEqual(
                population.frame,
                "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000_OSCULATING",
            )
            self.assertEqual(_sha256(path), population.source_artifact_sha256)
            selected = _selected(_rows(path), unnamed=unnamed)
            self.assertEqual(
                [member.observed_source.spk_id for member in population.members],
                [int(row["spkid"]) for row in selected],
            )
            for member, row in zip(population.members, selected, strict=True):
                self.assertEqual(float(row["epoch"]), DE441_CONTROL_EPOCH_JD_TDB)
                self.assertEqual(row["equinox"], "J2000")
                self.assertEqual(member.semimajor_axis_AU, float(row["a"]))
                self.assertEqual(member.eccentricity, float(row["e"]))
                self.assertEqual(member.inclination_rad, math.radians(float(row["i"])))
                self.assertEqual(
                    member.longitude_of_ascending_node_rad,
                    math.radians(float(row["om"])),
                )
                self.assertEqual(
                    member.argument_of_periapsis_rad,
                    math.radians(float(row["w"])),
                )
                self.assertEqual(member.mean_anomaly_rad, math.radians(float(row["ma"])))
                source = member.observed_source
                self.assertEqual(source.primary_designation, str(row["pdes"]))
                self.assertEqual(source.full_name, str(row["full_name"]).strip())
                self.assertEqual(source.orbit_solution_id, str(row["orbit_id"]))
                self.assertEqual(source.data_arc_days, int(row["data_arc"]))
                self.assertEqual(source.observations_used, int(row["n_obs_used"]))
                self.assertEqual(source.sigma_semimajor_axis_AU, float(row["sigma_a"]))
                self.assertEqual(source.sigma_eccentricity, float(row["sigma_e"]))
                self.assertEqual(source.sigma_inclination_rad, math.radians(float(row["sigma_i"])))

    def test_perturbations_are_deterministic_and_inside_each_source_sigma_box(self) -> None:
        for population, spec in (
            (DETACHED_TNO_SCREEN_V1, DETACHED_TNO_SENSITIVITY_V1),
            (SCATTERING_DISK_CONTROL_V1, SCATTERING_DISK_SENSITIVITY_V1),
        ):
            first = materialize_perturbation_ensemble(spec, population)
            second = materialize_perturbation_ensemble(spec, population)
            self.assertEqual(first.content_sha256, second.content_sha256)
            self.assertEqual(len(first.members), len(population.members) * 8)
            parents = {member.object_id: member for member in population.members}
            for realization in first.members:
                parent = parents[realization.parent_object_id]
                for name, offset in realization.offsets:
                    maximum = 3.0 * parent.observed_source.sigma_for(name)
                    self.assertLess(abs(offset), maximum)

    def test_claims_and_source_bindings_fail_closed(self) -> None:
        self.assertRegex(SCIENCE_INPUTS_CONTENT_SHA256, r"^[0-9a-f]{64}$")
        self.assertFalse(any(NON_INFERENCE_STATUS.values()))
        self.assertIs(
            get_population(DETACHED_TNO_SCREEN_V1.population_id),
            DETACHED_TNO_SCREEN_V1,
        )
        forged = replace(
            DETACHED_TNO_SENSITIVITY_V1,
            population_sha256="0" * 64,
        )
        with self.assertRaises(ScienceInputContractError):
            materialize_perturbation_ensemble(forged, DETACHED_TNO_SCREEN_V1)
        with self.assertRaises(ScienceInputContractError):
            replace(DETACHED_TNO_SCREEN_V1, claim_ceiling="SCIENCE_RESULT")


if __name__ == "__main__":
    unittest.main()
