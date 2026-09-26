from __future__ import annotations

import unittest

import numpy as np

from jxplanetx.gr15_eih_1pn_batch import (
    CUDA_BACKEND_ID,
    integrate_gr15_eih_1pn_batch,
    integrate_gr15_eih_1pn_cpu_batch,
)
from tests.test_gr15_eih_1pn_batch import _fixture, _parameters, _spec


def _require_cuda(test: unittest.TestCase) -> None:
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            test.skipTest("CUDA device unavailable")
    except Exception as exc:
        test.skipTest(f"CUDA runtime unavailable: {exc}")


class GR15EIH1PNBatchCUDAHardwareTests(unittest.TestCase):
    def test_forced_cuda_returns_numpy_and_matches_cpu(self) -> None:
        _require_cuda(self)
        positions, velocities, gm = _fixture(4)
        spec = _spec()
        parameters = _parameters()
        expected = integrate_gr15_eih_1pn_cpu_batch(
            positions, velocities, gm, spec, parameters, workers=4
        )
        actual = integrate_gr15_eih_1pn_batch(
            positions,
            velocities,
            gm,
            spec,
            parameters,
            backend="cuda",
        )
        self.assertEqual(actual.decision.backend_id, CUDA_BACKEND_ID)
        self.assertEqual(actual.worker_count, 0)
        self.assertEqual(actual.kernel_launch_count, 1)
        self.assertGreaterEqual(actual.input_transfer_seconds, 0.0)
        self.assertGreaterEqual(actual.output_transfer_seconds, 0.0)
        self.assertIs(type(actual.checkpoint_positions_km), np.ndarray)
        np.testing.assert_allclose(
            actual.checkpoint_positions_km,
            expected.checkpoint_positions_km,
            rtol=2.0e-15,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            actual.checkpoint_velocities_km_s,
            expected.checkpoint_velocities_km_s,
            rtol=2.0e-14,
            atol=2.0e-16,
        )


if __name__ == "__main__":
    unittest.main()
