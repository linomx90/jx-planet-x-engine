import json
import tempfile
import unittest
from pathlib import Path

from jxplanetx.encounter_tail import (
    CHECKPOINT_SCHEMA,
    GridCell,
    _block_seed,
    _build_simulation,
    _cell_for,
    _grid,
    _load_latest_checkpoint,
    _write_checkpoint,
)

try:
    import rebound  # noqa: F401
except ImportError:
    rebound = None


def locked_design():
    return {
        "grid": {
            "a_AU": [100, 150, 225, 325, 450, 600, 800, 1000],
            "q0_AU": [35, 45, 60],
            "i_deg": [5, 20, 35],
        },
        "required_grid_cells": 72,
    }


def row(index, name, mass, x="0", y="0", z="0", vx="0", vy="0", vz="0"):
    return {
        "index": str(index), "name": name, "mass": str(mass),
        "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz,
    }


class EncounterTailDesignTests(unittest.TestCase):
    def test_locked_grid_has_72_bound_cells(self):
        cells = _grid(locked_design())
        self.assertEqual(len(cells), 72)
        self.assertTrue(all(0 <= cell.e < 1 for cell in cells))
        self.assertEqual((cells[0].a_AU, cells[0].q0_AU, cells[0].i_deg), (100, 35, 5))
        self.assertEqual((cells[-1].a_AU, cells[-1].q0_AU, cells[-1].i_deg), (1000, 60, 35))

    def test_stratification_counts_are_locked(self):
        cells = _grid(locked_design())
        full_counts = [0] * 72
        for block in range(10):
            block_counts = [0] * 72
            for local in range(1000):
                index = _cell_for(cells, block, local, 8).index
                block_counts[index] += 1
                full_counts[index] += 1
            self.assertEqual(set(block_counts), {13, 14})
        self.assertEqual(set(full_counts), {138, 139})

    def test_block_seeds_are_reproducible_and_independent(self):
        self.assertEqual(_block_seed("locked", 4), _block_seed("locked", 4))
        self.assertNotEqual(_block_seed("locked", 4), _block_seed("locked", 5))


@unittest.skipUnless(rebound is not None, "optional REBOUND backend is not installed")
class EncounterTailCheckpointTests(unittest.TestCase):
    def test_mercurius_checkpoint_replay_is_exact(self):
        rows = [
            row(0, "Sun", 1),
            row(1, "Jupiter", 0.001, x="5", vy="2.8"),
            row(2, "Saturn", 0.0003, x="9", vy="2.1"),
            row(3, "Uranus", 0.00004, x="19", vy="1.44"),
            row(4, "Neptune", 0.00005, x="30", vy="1.15"),
        ]
        cells = [GridCell(0, 100, 35, 5)]
        simulation, metadata, _ = _build_simulation(rows, cells, 0, [0], "locked", 8, 0.0625)
        simulation.integrate(0.25, exact_finish_time=1)
        tracker = {"minimum_q": [metadata[0]["q0_AU"]], "marker": "sidecar"}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertTrue(_write_checkpoint(directory, simulation, tracker, 1, "job"))
            loaded = _load_latest_checkpoint(directory, "job")
            self.assertIsNotNone(loaded)
            replay, replay_tracker, index = loaded
            self.assertEqual(index, 1)
            self.assertEqual(replay_tracker, tracker)
            self.assertEqual(replay.N, simulation.N)
            self.assertEqual(replay.N_active, simulation.N_active)
            self.assertEqual(replay.ri_mercurius.r_crit_hill, 3.0)
            state = json.loads((directory / "checkpoint_0001.json").read_text())
            self.assertEqual(state["schema"], CHECKPOINT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
