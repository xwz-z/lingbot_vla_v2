import unittest

import numpy as np

from lingbotvla.utils.normalize import RunningStats


class RunningStatsPrecisionTest(unittest.TestCase):
    def test_preserves_small_variance_around_one_from_float32_input(self):
        values = np.linspace(0.9998, 1.0002, 20_000, dtype=np.float32)[:, None]
        stats = RunningStats()

        for chunk in np.array_split(values, 20):
            stats.update(chunk)

        result = stats.get_statistics()
        expected = values.astype(np.float64)
        self.assertGreater(result.std[0], 0.0)
        np.testing.assert_allclose(result.mean, expected.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(result.std, expected.std(axis=0), rtol=1e-6, atol=1e-12)

    def test_large_repeated_float32_batches_do_not_drift_constant_mean(self):
        values = np.full((80_000, 1), np.float32(5.4), dtype=np.float32)
        stats = RunningStats()

        for _ in range(10):
            stats.update(values)

        result = stats.get_statistics()
        expected = float(np.float32(5.4))
        self.assertAlmostEqual(float(result.mean[0]), expected, places=12)
        self.assertLessEqual(float(result.std[0]), 1e-7)


if __name__ == "__main__":
    unittest.main()
