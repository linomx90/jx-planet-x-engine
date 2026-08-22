from __future__ import annotations

import unittest
from pathlib import Path

from jxplanetx.de441_population import (
    _build_simulation,
    _load_population,
    _paired_effect,
    _state_rows,
    _verify_matched_states,
)


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs" / "de441_population_100k"


class De441PopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.population = _load_population(RUN / "population_elements_v1.csv")

    def test_locked_population_is_complete_and_injection_eligible(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
