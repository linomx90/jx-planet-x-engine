from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jxplanetx.independent_dop853 import (
    DOP853,
    _elements_to_cartesian,
    _load_checkpoint,
    _load_population,
    _orbital_elements,
    _rhs_factory,
    _wasserstein,
    _write_checkpoint,
)


PROJECT = Path(__file__).resolve().parents[1]
POPULATION_RUN = PROJECT / "runs" / "de441_population_100k"
INDEPENDENT_RUN = PROJECT / "runs" / "independent_dop853_10k"


class IndependentDop853Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.population = _load_population(POPULATION_RUN / "population_elements_v1.csv")

    def test_element_conversion_roundtrip(self) -> None:
        elements = self.population[0][:10]
        gm = 39.4771623747117388
        position, velocity = _elements_to_cartesian(
            elements, np.zeros(3), np.zeros(3), gm
        )
        state = np.concatenate((np.vstack((np.zeros((1, 3)), position)).ravel(),
                                np.vstack((np.zeros((1, 3)), velocity)).ravel()))
        q, inclination, bound = _orbital_elements(state, 1, gm)
        self.assertTrue(np.all(bound))
        self.assertLess(float(np.max(np.abs(q - [row["q0_AU"] for row in elements]))), 1e-10)
        self.assertLess(
            float(np.max(np.abs(inclination - [row["i0_deg"] for row in elements]))),
            1e-10,
        )

    def test_independent_two_body_period(self) -> None:
        masses = np.array([4.0 * math.pi * math.pi])
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 2.0 * math.pi, 0.0])
        solver = DOP853(
            _rhs_factory(masses, 1),
            0.0,
            state,
            1.0,
            rtol=1e-12,
            atol=1e-14,
            max_step=0.01,
        )
        while solver.status == "running":
            solver.step()
        final_position = solver.y[3:6]
        final_velocity = solver.y[9:12]
        self.assertLess(float(np.linalg.norm(final_position - [1.0, 0.0, 0.0])), 1e-10)
        self.assertLess(
            float(np.linalg.norm(final_velocity - [0.0, 2.0 * math.pi, 0.0])), 1e-9
        )

    def test_checkpoint_replay_is_binary64_exact(self) -> None:
        state = np.linspace(-1.0, 1.0, 24)
        tracker = {
            "minimum_q": np.array([31.0, 32.0]),
            "first_low_q_year": [None, None],
            "ever_unbound": np.array([False, False]),
            "first_unbound_year": [None, None],
            "sample_count": 1,
        }
        segments = [{"segment_index": 1, "accepted_steps": 10}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(_write_checkpoint(root, 0, 0.0, state, tracker, segments, "job"))
            loaded = _load_checkpoint(root, "job")
            self.assertIsNotNone(loaded)
            replay, replay_tracker, replay_segments, index, time_year = loaded  # type: ignore[misc]
            self.assertTrue(np.array_equal(state, replay))
            self.assertTrue(np.array_equal(tracker["minimum_q"], replay_tracker["minimum_q"]))
            self.assertEqual(replay_segments, segments)
            self.assertEqual(index, 0)
            self.assertEqual(time_year, 0.0)

    def test_unequal_population_wasserstein(self) -> None:
        self.assertAlmostEqual(_wasserstein([0.0, 2.0], [1.0]), 1.0)

    def test_selection_is_hash_locked_and_unique(self) -> None:
        selection = json.loads((INDEPENDENT_RUN / "selection_v1.json").read_text())
        self.assertEqual(selection["selected_blocks"], [3, 9, 33, 47, 49, 64, 65, 74, 83, 98])
        self.assertEqual(len(set(selection["selected_blocks"])), 10)
        self.assertEqual(selection["selection_status"], "OUTCOME_BLIND_HASH_RANKED")

    def test_module_does_not_import_rebound(self) -> None:
        source = (PROJECT / "src" / "jxplanetx" / "independent_dop853.py").read_text()
        self.assertNotIn("import rebound", source.lower())


if __name__ == "__main__":
    unittest.main()
