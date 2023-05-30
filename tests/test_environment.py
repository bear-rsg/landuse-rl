import unittest
import numpy as np
import pandas as pd

from src.environment import Environment


class MyTestCase(unittest.TestCase):
    def expectEqual(self, first, second, msg=None):
        with self.subTest():
            self.assertEqual(first, second, msg)

    def expectSequenceEqual(self, first, second, msg=None):
        with self.subTest():
            self.assertSequenceEqual(first, second, msg)

    def expectAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        with self.subTest():
            self.assertAlmostEqual(first, second, places, msg, delta)

    def expectTrue(self, expr, msg=None):
        with self.subTest():
            self.assertTrue(expr, msg)

    def expectFalse(self, expr, msg=None):
        with self.subTest():
            self.assertFalse(expr, msg)

    def expectRaises(self, exception, callable, *args, **kwds):
        with self.subTest():
            self.assertRaises(exception, callable, *args, **kwds)


class TestVariables(MyTestCase):
    def test_initial_variables_1(self):
        env = Environment(sig_type_interval=(0, 15),
                          sig_type_step=1,
                          land_use_interval=(-1, 1),
                          land_use_n_points=21,
                          greenspace_interval=(0, 1),
                          greenspace_n_points=11,
                          job_type_interval=(0, 1),
                          job_type_n_points=11,
                          epsilon=0.001,
                          seed=1,
                          debug=True)

        data = {'signature_type': np.array([10, 8, 5], dtype=np.int64),
                'use': np.array([-0.999, 0.999, 0], dtype=np.float64),
                'greenspace': np.array([0, 0.999, 0.999], dtype=np.float64),
                'job_types': np.array([0.4002, 0.1008, 0.999], dtype=np.float64)}
        expected = pd.DataFrame(data)

        for i in range(env.variables.shape[0]):
            for j in range(env.variables.shape[1]):
                self.expectAlmostEqual(env.variables.iloc[i, j], expected.iloc[i, j], places=5, msg=f"i={i},j={j}")

    def test_initial_variables_2(self):
        env = Environment(sig_type_interval=(0, 15),
                          sig_type_step=1,
                          land_use_interval=(-1, 1),
                          land_use_n_points=11,
                          greenspace_interval=(0, 1),
                          greenspace_n_points=7,
                          job_type_interval=(0, 1),
                          job_type_n_points=5,
                          epsilon=0.001,
                          seed=1,
                          debug=True)

        data = {'signature_type': np.array([10, 8, 5], dtype=np.int64),
                'use': np.array([-0.999, 0.999, 0], dtype=np.float64),
                'greenspace': np.array([0, 0.999, 0.999], dtype=np.float64),
                'job_types': np.array([0.5, 0.001, 0.999], dtype=np.float64)}
        expected = pd.DataFrame(data)

        for i in range(env.variables.shape[0]):
            for j in range(env.variables.shape[1]):
                self.expectAlmostEqual(env.variables.iloc[i, j], expected.iloc[i, j], places=5, msg=f"i={i},j={j}")


if __name__ == '__main__':
    unittest.main()
