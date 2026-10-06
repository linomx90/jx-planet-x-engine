from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from jxplanetx.de441_population import (
    BLOCK_SCHEMA,
    COMMON_NAMES,
    POPULATION_COLUMNS,
    TRACER_COLUMNS,
    _block_content_sha256,
    _block_lock,
    _build_simulation,
    _cached_sha256,
    _decode_json_object,
    _job_sha256,
    _load_population,
    _paired_effect,
    _run_block,
    _state_rows,
    _validate_cached_block,
    _validate_job,
    _validate_tracer_csv,
    _validate_tracker,
    _verify_matched_states,
)


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs" / "de441_population_100k"


class De441PopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.population = _load_population(RUN / "population_elements_v1.csv")

    def test_locked_population_is_complete_and_injection_eligible(self) -> None:
        population_path = RUN / "population_elements_v1.csv"
        manifest = json.loads((RUN / "population_manifest_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(population_path.read_bytes()).hexdigest(),
            manifest["population_sha256"],
        )
        self.assertEqual(len(self.population), 100)
        self.assertTrue(all(len(rows) == 1000 for rows in self.population.values()))
        rows = [row for block in self.population.values() for row in block]
        self.assertEqual(len({row["logical_id"] for row in rows}), 100000)
        self.assertTrue(all(row["a0_AU"] > row["q0_AU"] > 30.0 for row in rows))

    def test_source_control_common_state_is_exact(self) -> None:
        source = _state_rows(RUN / "states" / "de441_source_9118_state.csv")
        control = _state_rows(RUN / "states" / "de441_control_state.csv")
        audit = _verify_matched_states(source, control)
        self.assertEqual(audit["maximum_common_relative_state_difference"], 0.0)
        self.assertEqual(audit["source_only_body"], "P9_BB21_idx9118")

    def test_retained_evidence_status_matches_this_checkout(self) -> None:
        status = json.loads(
            (RUN / "retained_evidence_status_v1.json").read_text(encoding="utf-8")
        )
        observation = status["checkout_observation"]
        self.assertFalse(observation["execution_directories_present"])
        self.assertEqual(observation["block_summaries"], 0)
        self.assertEqual(observation["tracer_csvs"], 0)
        self.assertEqual(observation["checkpoint_archives"], 0)
        self.assertFalse(
            any(
                path
                for pattern in ("summary.json", "tracers.csv", "checkpoint_*.bin")
                for path in RUN.rglob(pattern)
            )
        )
        provenance = status["candidate_9118_provenance"]
        self.assertFalse(provenance["declared_metadata_artifact_retained"])
        self.assertEqual(
            provenance["declared_candidate_metadata_sha256"],
            "509917e0093107464d9ee45ed2c8e9f403403b2bb0e94455fa3614825917f8b0",
        )
        retirement = provenance["governing_later_record"]
        retirement_path = (RUN / retirement["path"]).resolve()
        self.assertEqual(hashlib.sha256(retirement_path.read_bytes()).hexdigest(), retirement["sha256"])

    def test_gm_unit_simulation_reconstructs_elements(self) -> None:
        try:
            import rebound  # noqa: F401
        except ImportError:
            self.skipTest("optional REBOUND backend is not installed")
        control = _state_rows(RUN / "states" / "de441_control_state.csv")
        simulation, names = _build_simulation(control, self.population[0][:3], 0.0625)
        self.assertEqual(simulation.G, 1.0)
        self.assertEqual(simulation.N_active, len(names))
        for offset, element in enumerate(self.population[0][:3]):
            orbit = simulation.particles[simulation.N_active + offset].orbit(
                primary=simulation.particles[0]
            )
            self.assertAlmostEqual(orbit.a, element["a0_AU"], places=10)
            self.assertAlmostEqual(orbit.a * (1.0 - orbit.e), element["q0_AU"], places=10)

    def test_identical_paired_outputs_classify_as_equivalent(self) -> None:
        rows = []
        for block in range(2):
            for local in range(2):
                element = self.population[block][local]
                rows.append(
                    {
                        **{key: str(value) for key, value in element.items()},
                        "minimum_sampled_q_AU": str(element["q0_AU"]),
                        "sampled_injection": "0",
                        "ever_unbound_at_sample": "0",
                        "bound_final": "1",
                        "final_q_AU": str(element["q0_AU"]),
                        "final_i_deg": str(element["i0_deg"]),
                        "minimum_sampled_neptune_hill_ratio": "10",
                        "minimum_sampled_source_hill_ratio": "",
                    }
                )
        result = _paired_effect(rows, [dict(row) for row in rows], "test", 99, 0.001)
        self.assertEqual(
            result["source_minus_control"]["effect_classification"],
            "EQUIVALENT_WITHIN_LOCKED_MARGIN",
        )
        self.assertEqual(result["source_minus_control"]["sampled_injection_fraction"], 0.0)


class De441PopulationCustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.element = _load_population(RUN / "population_elements_v1.csv")[0][0]

    def _job(self, root: Path, *, arm: str = "control") -> dict[str, object]:
        core: dict[str, object] = {
            "contract_sha256": "0" * 64,
            "arm": arm,
            "stage": "custody-test",
            "state_csv": str(root / "state.csv"),
            "state_sha256": "1" * 64,
            "population_csv": str(root / "population.csv"),
            "population_sha256": "2" * 64,
            "block_index": 0,
            "local_count": 1,
            "dt_years": 0.5,
            "sample_years": 1.0,
            "duration_years": 2.0,
            "aggregate_years": 1.0,
            "checkpoint_years": 1.0,
            "total_samples": 2,
            "aggregate_stride_samples": 1,
            "checkpoint_stride_samples": 1,
            "q_lower": 29.999999,
        }
        return {
            **core,
            "job_sha256": _job_sha256(core),
            "directory": str(root / "block"),
        }

    def _tracer_row(self) -> dict[str, object]:
        element = self.element
        return {
            **element,
            "minimum_sampled_q_AU": element["q0_AU"],
            "sampled_injection": 0,
            "first_sampled_low_q_year": None,
            "ever_unbound_at_sample": 0,
            "first_unbound_year": None,
            "bound_final": 1,
            "final_q_AU": element["q0_AU"],
            "final_i_deg": element["i0_deg"],
            "minimum_sampled_neptune_distance_AU": 1.0,
            "minimum_sampled_neptune_hill_ratio": 2.0,
            "minimum_sampled_source_distance_AU": None,
            "minimum_sampled_source_hill_ratio": None,
        }

    def _write_tracers(self, path: Path, row: dict[str, object]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACER_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)

    def _timeseries(self) -> list[dict[str, object]]:
        return [
            {
                "time_year": float(index),
                "bound": 1,
                "survival_fraction": 1.0,
                "cumulative_sampled_injections": 0,
                "cumulative_sampled_injection_fraction": 0.0,
                "mean_current_bound_q_AU": self.element["q0_AU"],
                "current_bound_inclination_width_deg": 0.0,
            }
            for index in range(3)
        ]

    def _summary(self, job: dict[str, object], tracer_digest: str) -> dict[str, object]:
        active = [
            {
                "index": str(index),
                "m": float(1.0).hex(),
                "x": float(index).hex(),
                "y": 0.0.hex(),
                "z": 0.0.hex(),
                "vx": 0.0.hex(),
                "vy": 0.0.hex(),
                "vz": 0.0.hex(),
            }
            for index in range(len(COMMON_NAMES))
        ]
        from jxplanetx.population_scale import _canonical_sha256

        summary: dict[str, object] = {
            "schema": BLOCK_SCHEMA,
            "job_sha256": job["job_sha256"],
            "arm": job["arm"],
            "stage": job["stage"],
            "block_index": job["block_index"],
            "tracers": 1,
            "particles": len(COMMON_NAMES) + 1,
            "N_active": len(COMMON_NAMES),
            "dt_years": job["dt_years"],
            "duration_years": job["duration_years"],
            "sample_years": job["sample_years"],
            "sample_count_including_t0": 3,
            "checkpoint_epochs_including_t0": 3,
            "restart_replay_state_hash_exact": True,
            "cartesian_state_finite": True,
            "active_endpoint_state": active,
            "active_endpoint_state_sha256": _canonical_sha256(active),
            "tracer_csv": "tracers.csv",
            "tracer_csv_sha256": tracer_digest,
            "summary_json": "summary.json",
            "timeseries": self._timeseries(),
            "elapsed_seconds": 1.0,
        }
        summary["block_content_sha256"] = _block_content_sha256(summary)
        return summary

    def _tracker(self) -> dict[str, object]:
        return {
            "minimum_q": [self.element["q0_AU"]],
            "first_low_q_year": [None],
            "ever_unbound": [False],
            "first_unbound_year": [None],
            "minimum_neptune_distance_AU": [1.0],
            "minimum_neptune_hill_ratio": [2.0],
            "minimum_source_distance_AU": [None],
            "minimum_source_hill_ratio": [None],
            "neptune_index": COMMON_NAMES.index("Neptune"),
            "source_index": None,
            "neptune_hill_radius_AU": 0.5,
            "source_hill_radius_AU": None,
            "sample_count": 2,
            "timeseries": self._timeseries()[:2],
        }

    def test_job_digest_is_path_independent_and_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = self._job(Path(temporary))
            second = dict(first)
            second["state_csv"] = "/relocated/state.csv"
            second["population_csv"] = "/relocated/population.csv"
            second["directory"] = "/relocated/block"
            self.assertEqual(_job_sha256(first), _job_sha256(second))
            _validate_job(second)
            second["duration_years"] = 3.0
            with self.assertRaises(ValueError):
                _validate_job(second)

    def test_json_custody_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            _decode_json_object(b'{"schema":"one","schema":"two"}', "test record")
        with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
            _decode_json_object(b'{"value":NaN}', "test record")

    def test_population_and_hash_caches_follow_file_identity_not_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "population.csv"
            row = {key: self.element[key] for key in POPULATION_COLUMNS}
            self._write_population(path, row)
            first_digest = _cached_sha256(path)
            first = _load_population(path)[0][0]["q0_AU"]
            changed = dict(row)
            changed_q = float(row["q0_AU"]) + 1e-6
            changed["q0_AU"] = changed_q
            changed["e0"] = 1.0 - changed_q / float(row["a0_AU"])
            replacement = path.with_suffix(".replacement")
            self._write_population(replacement, changed)
            os.replace(replacement, path)
            self.assertNotEqual(_cached_sha256(path), first_digest)
            self.assertNotEqual(_load_population(path)[0][0]["q0_AU"], first)

    def _write_population(self, path: Path, row: dict[str, object]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=POPULATION_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)

    def test_full_tracer_validation_rejects_self_consistent_fabrication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            block = root / "block"
            block.mkdir()
            job = self._job(root)
            tracer = block / "tracers.csv"
            row = self._tracer_row()
            self._write_tracers(tracer, row)
            _validate_tracer_csv(tracer, job, [self.element])
            digest = hashlib.sha256(tracer.read_bytes()).hexdigest()
            summary = self._summary(job, digest)
            (block / "summary.json").write_text(
                json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
            )
            _validate_cached_block(block, job, [self.element], len(COMMON_NAMES))

            wrong_type = dict(summary)
            wrong_type["duration_years"] = 2
            wrong_type["block_content_sha256"] = _block_content_sha256(wrong_type)
            (block / "summary.json").write_text(
                json.dumps(wrong_type, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "floating-point"):
                _validate_cached_block(block, job, [self.element], len(COMMON_NAMES))

            row["sampled_injection"] = 1
            self._write_tracers(tracer, row)
            summary["tracer_csv_sha256"] = hashlib.sha256(tracer.read_bytes()).hexdigest()
            summary["block_content_sha256"] = _block_content_sha256(summary)
            (block / "summary.json").write_text(
                json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "injection flag"):
                _validate_cached_block(block, job, [self.element], len(COMMON_NAMES))

    def test_checkpoint_tracker_is_bound_to_index_time_and_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = self._job(Path(temporary))
            tracker = self._tracker()
            _validate_tracker(tracker, job, [self.element], list(COMMON_NAMES), 1)
            broken = copy.deepcopy(tracker)
            broken["sample_count"] = 1
            with self.assertRaisesRegex(ValueError, "sample_count"):
                _validate_tracker(broken, job, [self.element], list(COMMON_NAMES), 1)
            broken = copy.deepcopy(tracker)
            broken["timeseries"][1]["time_year"] = 1.5
            with self.assertRaisesRegex(ValueError, "wrong epoch"):
                _validate_tracker(broken, job, [self.element], list(COMMON_NAMES), 1)

    def test_block_lock_refuses_a_second_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with _block_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "another worker"):
                    with _block_lock(directory):
                        self.fail("second writer unexpectedly acquired the block lock")

    def test_real_block_resume_replays_checkpoint_chain_and_rejects_time_tampering(self) -> None:
        try:
            import rebound  # noqa: F401
        except ImportError:
            self.skipTest("optional REBOUND backend is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = RUN / "states" / "de441_control_state.csv"
            population = RUN / "population_elements_v1.csv"
            core: dict[str, object] = {
                "contract_sha256": "0" * 64,
                "arm": "control",
                "stage": "custody-smoke",
                "state_csv": str(state),
                "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                "population_csv": str(population),
                "population_sha256": hashlib.sha256(population.read_bytes()).hexdigest(),
                "block_index": 0,
                "local_count": 1,
                "dt_years": 0.0625,
                "sample_years": 0.25,
                "duration_years": 0.5,
                "aggregate_years": 0.25,
                "checkpoint_years": 0.25,
                "total_samples": 2,
                "aggregate_stride_samples": 1,
                "checkpoint_stride_samples": 1,
                "q_lower": 29.999999,
            }
            job = {
                **core,
                "job_sha256": _job_sha256(core),
                "directory": str(root / "block"),
            }
            first = _run_block(job)
            self.assertNotIn("resumed_from_complete_block", first)
            second = _run_block(job)
            self.assertTrue(second["resumed_from_complete_block"])
            checkpoint = root / "block" / "checkpoint_001.json"
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            duplicate = '{"schema":"duplicate",' + json.dumps(record, sort_keys=True)[1:]
            checkpoint.write_text(duplicate + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
                _run_block(job)
            record["time_year"] = 0.125
            checkpoint.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "time mismatch"):
                _run_block(job)


if __name__ == "__main__":
    unittest.main()
