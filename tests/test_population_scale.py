import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from jxplanetx.population_scale import (
    CORE_NAMES,
    OrbitShape,
    _open_uniform,
    _orbit_shape,
    _phase_angles,
    _population_metrics,
    _verify_matched_states,
    run_population_scale_gate,
)
from jxplanetx.provenance import runtime_source_manifest, sha256_file

try:
    import rebound
except ImportError:
    rebound = None


def row(index, name, mass, x="0", y="0", z="0", vx="0", vy="0", vz="0"):
    return {
        "index": str(index),
        "name": name,
        "mass": str(mass),
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
    }


def matched_fixture():
    control = [row(index, name, "1" if name == "Sun" else "0.001") for index, name in enumerate(CORE_NAMES)]
    control.append(row(5, "t000", "0", x="1", vy=str(2.0 * math.pi)))
    source = [dict(item) for item in control[:5]]
    source.append(row(5, "P9", "0.00001", x="500", vy="0.1"))
    tracer = dict(control[5])
    tracer["index"] = "6"
    source.append(tracer)
    return source, control


class PopulationScaleTests(unittest.TestCase):
    def test_counter_uniform_is_open_and_reproducible(self):
        first = _open_uniform("seed", 123, "M")
        self.assertEqual(first, _open_uniform("seed", 123, "M"))
        self.assertGreater(first, 0.0)
        self.assertLess(first, 1.0)
        self.assertNotEqual(first, _open_uniform("seed", 124, "M"))

    def test_phase_angles_are_deterministic_and_in_range(self):
        values = _phase_angles("locked", 99999)
        self.assertEqual(values, _phase_angles("locked", 99999))
        self.assertTrue(all(0.0 < value < 2.0 * math.pi for value in values))

    def test_exact_heliocentric_match_and_source_detection(self):
        source, control = matched_fixture()
        result = _verify_matched_states(source, control)
        self.assertEqual(result["source_name"], "P9")
        self.assertEqual(result["template_count"], 1)
        self.assertEqual(len(result["control_relative_state_sha256"]), 64)

    def test_relative_state_mismatch_is_rejected(self):
        source, control = matched_fixture()
        source[-1]["x"] = "1.0000001"
        with self.assertRaisesRegex(ValueError, "relative-state mismatch"):
            _verify_matched_states(source, control)

    def test_circular_template_shape(self):
        source, control = matched_fixture()
        shape = _orbit_shape(control[-1], control[0])
        self.assertAlmostEqual(shape.a_AU, 1.0, places=12)
        self.assertAlmostEqual(shape.e, 0.0, places=12)
        self.assertAlmostEqual(shape.i_rad, 0.0, places=12)
        self.assertAlmostEqual(shape.q_AU, 1.0, places=12)

    def test_q_hysteresis_excludes_boundary_jitter_and_uses_eligible_denominator(self):
        class FakeOrbit:
            def __init__(self, q):
                self.a = 100.0
                self.e = 1.0 - q / self.a
                self.inc = 0.1

        class FakeParticle:
            def __init__(self, q):
                self._orbit = FakeOrbit(q)

            def orbit(self, primary):
                return self._orbit

        class FakeSimulation:
            particles = [object(), FakeParticle(29.99999995), FakeParticle(29.0), FakeParticle(29.0)]

        templates = [
            OrbitShape("boundary", 100.0, 0.7, 0.1, 30.00000005),
            OrbitShape("high", 100.0, 0.69, 0.1, 31.0),
            OrbitShape("low", 100.0, 0.71, 0.1, 29.0),
        ]
        metrics, _, _, replicates = _population_metrics(
            FakeSimulation(), 1, templates, 3, 30.0, 1e-6, 3
        )
        self.assertEqual(metrics["q_below_lower_band_final"], 2)
        self.assertEqual(metrics["q_boundary_final"], 1)
        self.assertEqual(metrics["initially_high_q_eligible"], 1)
        self.assertEqual(metrics["endpoint_injections"], 1)
        self.assertEqual(metrics["endpoint_injection_fraction_of_eligible"], 1.0)
        self.assertEqual(replicates[0]["q_boundary_fraction"], 1 / 3)


@unittest.skipUnless(rebound is not None, "optional REBOUND backend is not installed")
class PopulationScaleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.control = self.root / "control.csv"
        self.source = self.root / "source.csv"
        self.wheel = self.root / "rebound.whl"
        self.wheel.write_bytes(b"locked test wheel fixture")
        circular = [
            ("Sun", 1.0, 0.0),
            ("Jupiter", 0.00095, 5.2),
            ("Saturn", 0.00028, 9.5),
            ("Uranus", 0.000044, 19.2),
            ("Neptune", 0.000052, 30.1),
        ]
        control_rows = []
        for index, (name, mass, radius) in enumerate(circular):
            speed = 0.0 if radius == 0.0 else math.sqrt(4.0 * math.pi * math.pi / radius)
            control_rows.append(row(index, name, mass, x=repr(radius), vy=repr(speed)))
        control_rows.append(row(5, "t000", 0.0, x="2", vy=repr(math.sqrt(4.0 * math.pi * math.pi / 2.0))))
        source_rows = [dict(item) for item in control_rows[:5]]
        source_rows.append(row(5, "P9", 0.00001, x="100", vy=repr(math.sqrt(4.0 * math.pi * math.pi / 100.0))))
        source_tracer = dict(control_rows[-1])
        source_tracer["index"] = "6"
        source_rows.append(source_tracer)
        for path, rows in ((self.control, control_rows), (self.source, source_rows)):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("index", "name", "mass", "x", "y", "z", "vx", "vy", "vz"))
                writer.writeheader()
                writer.writerows(rows)

    def tearDown(self):
        self.temporary.cleanup()

    def contract(self, rebound_version=None):
        manifest = runtime_source_manifest()
        binary = Path(rebound.clibrebound._name)
        return {
            "schema": "jx-population-scale-contract/v1",
            "runner_source_tree_sha256": manifest["tree_sha256"],
            "source_state_csv": str(self.source),
            "control_state_csv": str(self.control),
            "state_sha256": {"source": sha256_file(self.source), "control": sha256_file(self.control)},
            "archive_match_tolerances": {
                "position_AU": "0",
                "velocity_AU_per_year": "0",
                "mass_Msun": "0"
            },
            "population_design": {
                "seed": "integration-test",
                "seed_blocks": 1,
                "replicates_per_block": 1,
                "tracers_per_replicate": 10,
                "phase_generator": "sha256-counter-open-uniform/v1",
            },
            "dynamics": {
                "integrator": "mercurius",
                "rebound_version": rebound_version or rebound.__version__,
                "rebound_build": rebound.__build__,
                "rebound_binary_sha256": sha256_file(binary),
                "rebound_wheel_file": str(self.wheel),
                "rebound_wheel_sha256": sha256_file(self.wheel),
                "testparticle_type": 0,
                "dt_years": 0.01,
                "gate_years": 0.02,
                "target_years": 0.02,
                "energy_check_interval_years": 0.01,
            },
            "q_threshold_AU": 3.0,
            "q_hysteresis_AU": 1e-6,
            "operational_gates": {
                "max_relative_energy_drift": 1.0,
                "max_peak_rss_fraction": 0.99,
                "max_projected_paired_hours": 12.0,
            },
            "projection_validation_status": "LOCKED_REPEAT_AND_LONG_HORIZON_TAIL_COMPLETE",
        }

    def test_tiny_locked_end_to_end_and_output_immutability(self):
        contract_path = self.root / "contract.json"
        output_path = self.root / "result.json"
        contract_path.write_text(json.dumps(self.contract(), indent=2) + "\n", encoding="utf-8")
        result = run_population_scale_gate(contract_path, output_path)
        self.assertEqual(result["verdict"], "SCALE_GATE_PASSED")
        self.assertEqual(result["design_counts"]["tracers_per_arm"], 10)
        self.assertTrue(result["operational_checks"]["massless_no_backreaction_configuration"])
        self.assertEqual(result["arms"]["control"]["active_only_twin_max_abs_state_difference"], 0.0)
        with self.assertRaises(FileExistsError):
            run_population_scale_gate(contract_path, output_path)

    def test_rebound_version_mismatch_fails_before_output(self):
        contract_path = self.root / "bad-contract.json"
        output_path = self.root / "bad-result.json"
        contract_path.write_text(json.dumps(self.contract("0.0.invalid"), indent=2) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "REBOUND runtime"):
            run_population_scale_gate(contract_path, output_path)
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
