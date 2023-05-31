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
    def setUp(self):
        data = {'air_quality': np.array([1, 1, 1], dtype=np.float64),
                'house_price': np.array([1, 1, 1], dtype=np.float64),
                'job_accessibility': np.array([1, 1, 1], dtype=np.float64),
                'greenspace_accessibility': np.array([1, 1, 1], dtype=np.float64)}
        self.target = pd.DataFrame(data)

    def test_initial_variables_1(self):
        env = Environment(self.target,
                          sig_type_interval=(0, 15),
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
        env = Environment(self.target,
                          sig_type_interval=(0, 15),
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


class TestIndicators(MyTestCase):
    def setUp(self):
        data = {'air_quality': np.array([1, 1, 1], dtype=np.float64),
                'house_price': np.array([1, 1, 1], dtype=np.float64),
                'job_accessibility': np.array([1, 1, 1], dtype=np.float64),
                'greenspace_accessibility': np.array([1, 1, 1], dtype=np.float64)}
        target = pd.DataFrame(data)

        self.env = Environment(target,
                               sig_type_interval=(0, 15),
                               sig_type_step=1,
                               land_use_interval=(-1, 1),
                               land_use_n_points=21,
                               greenspace_interval=(0, 1),
                               greenspace_n_points=11,
                               job_type_interval=(0, 1),
                               job_type_n_points=11,
                               epsilon=0.001,
                               max_air_quality=7.5,
                               max_house_price=0.5,
                               max_job_accessibility=0.5,
                               max_greenspace_accessibility=0.5,
                               seed=1,
                               debug=True)

    def test_initial_indicators_1(self):
        expected = np.array([0.666666, 0.999, 0.0, 0.4002, 0.533333, 0.999, 0.999, 0.1008, 0.333333, 0.0, 0.999, 0.999], dtype=np.float32)
        for i in range(expected.shape[0]):
            self.expectAlmostEqual(self.env.indicators[i], expected[i], places=5, msg=f"{i=}")


class TestActions(MyTestCase):
    def setUp(self):
        data = {'air_quality': np.array([1, 1, 1], dtype=np.float64),
                'house_price': np.array([1, 1, 1], dtype=np.float64),
                'job_accessibility': np.array([1, 1, 1], dtype=np.float64),
                'greenspace_accessibility': np.array([1, 1, 1], dtype=np.float64)}
        target = pd.DataFrame(data)

        self.env = Environment(target,
                               sig_type_interval=(0, 15),
                               sig_type_step=1,
                               land_use_interval=(-1, 1),
                               land_use_n_points=21,
                               greenspace_interval=(0, 1),
                               greenspace_n_points=11,
                               job_type_interval=(0, 1),
                               job_type_n_points=11,
                               epsilon=0.001,
                               max_air_quality=7.5,
                               max_house_price=0.5,
                               max_job_accessibility=0.5,
                               max_greenspace_accessibility=0.5,
                               seed=1,
                               debug=True)

    def test_initial_valid_actions_1(self):
        expected = np.array([0, 1, 2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 21, 23], dtype=np.int32)
        for i in range(self.env.valid_actions.shape[0]):
            self.expectEqual(self.env.valid_actions[i], expected[i], msg=f"{i=}")

    def test_step_raises(self):
        self.expectRaises(ValueError, self.env.step, 3)
        self.expectRaises(ValueError, self.env.step, 5)
        self.expectRaises(ValueError, self.env.step, 22)

    def test_step_1(self):
        # Step
        self.expectFalse(3 in self.env.valid_actions)
        self.env.step(2)
        data = {'signature_type': np.array([10, 8, 5], dtype=np.int64),
                'use': np.array([-0.8991, 0.999, 0], dtype=np.float64),
                'greenspace': np.array([0, 0.999, 0.999], dtype=np.float64),
                'job_types': np.array([0.4002, 0.1008, 0.999], dtype=np.float64)}
        expected = pd.DataFrame(data)
        for i in range(self.env.variables.shape[0]):
            for j in range(self.env.variables.shape[1]):
                self.expectAlmostEqual(self.env.variables.iloc[i, j], expected.iloc[i, j], places=5, msg=f"i={i},j={j}")

        # Step
        self.expectTrue(3 in self.env.valid_actions)
        self.env.step(3)
        expected.iloc[0, 1] = -0.999
        for i in range(self.env.variables.shape[0]):
            for j in range(self.env.variables.shape[1]):
                self.expectAlmostEqual(self.env.variables.iloc[i, j], expected.iloc[i, j], places=5, msg=f"i={i},j={j}")
        self.expectFalse(3 in self.env.valid_actions)

        # Step
        self.env.step(0)
        self.env.step(0)
        self.env.step(0)
        self.env.step(0)
        self.env.step(0)
        expected.iloc[0, 0] = 15
        for i in range(self.env.variables.shape[0]):
            for j in range(self.env.variables.shape[1]):
                self.expectAlmostEqual(self.env.variables.iloc[i, j], expected.iloc[i, j], places=5, msg=f"i={i},j={j}")
        self.expectFalse(0 in self.env.valid_actions)
        self.expectRaises(ValueError, self.env.step, 0)
        expected_indicators = np.array([1.0, 0.999, 0.0, 0.4002, 0.533333, 0.999, 0.999, 0.1008, 0.333333, 0.0, 0.999, 0.999], dtype=np.float32)
        for i in range(expected_indicators.shape[0]):
            self.expectAlmostEqual(self.env.indicators[i], expected_indicators[i], places=5, msg=f"{i=}")


class TestEpisode(MyTestCase):
    def setUp(self):
        data = {'air_quality': np.array([5.5, 4.0, 2.5], dtype=np.float64),
                'house_price': np.array([0.4995, 0.4995, 0.0], dtype=np.float64),
                'job_accessibility': np.array([0.0, 0.4995, 0.4995], dtype=np.float64),
                'greenspace_accessibility': np.array([0.2001, 0.0504, 0.4995], dtype=np.float64)}
        target = pd.DataFrame(data)

        self.env = Environment(target,
                               sig_type_interval=(0, 15),
                               sig_type_step=1,
                               land_use_interval=(-1, 1),
                               land_use_n_points=3,
                               greenspace_interval=(0, 1),
                               greenspace_n_points=2,
                               job_type_interval=(0, 1),
                               job_type_n_points=11,
                               epsilon=0.001,
                               max_air_quality=7.5,
                               max_house_price=0.5,
                               max_job_accessibility=0.5,
                               max_greenspace_accessibility=0.5,
                               tolerance=1e-5,
                               seed=1,
                               debug=True)

    def test_variable_values_1(self):
        expected = np.arange(0, 16, 1)
        for i in range(self.env.variable_values[0].shape[0]):
            self.expectAlmostEqual(self.env.variable_values[0][i], expected[i], places=1, msg=f"sig_type_values {i=}")

        expected = np.array([-0.999, 0.0, 0.999])
        for i in range(self.env.variable_values[1].shape[0]):
            self.expectAlmostEqual(self.env.variable_values[1][i], expected[i], places=5, msg=f"land_use_values {i=}")

        expected = np.array([0.0, 0.999])
        for i in range(self.env.variable_values[2].shape[0]):
            self.expectAlmostEqual(self.env.variable_values[2][i], expected[i], places=5, msg=f"greenspace_values {i=}")

        expected = np.array([0.001, 0.1008, 0.2006, 0.3004, 0.4002, 0.5, 0.5998, 0.6996, 0.7994, 0.8992, 0.999])
        for i in range(self.env.variable_values[3].shape[0]):
            self.expectAlmostEqual(self.env.variable_values[3][i], expected[i], places=5, msg=f"job_type_values {i=}")

    def test_episode_1(self):
        expected_indicators = np.array([0.6666667, 0.999,  0.0,        0.4002, 0.53333336, 0.999, \
                                        0.999    , 0.1008, 0.33333334, 0.0,    0.999,      0.999], dtype=np.float32)
        self.expectSequenceEqual(self.env.episode['indicators'][0].tolist(), expected_indicators.tolist(), msg=f"initial indicators")

        expected_state = np.array([-0.06666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.expectSequenceEqual(self.env.episode['state'][0].tolist(), expected_state.tolist(), msg=f"initial state")

        expected_norm = 0.066666
        self.expectAlmostEqual(self.env.episode['norm'][0], expected_norm, places=5, msg=f"initial norm")

        expected_done = 0
        self.expectEqual(self.env.episode['done'][0], expected_done, msg=f"initial done")

        # ----- Perform action ----- #
        self.env.step(0)
        # -------------------------- #

        expected_indicators = np.array([0.73333333, 0.999,  0.0,        0.4002, 0.53333336, 0.999, \
                                        0.999    , 0.1008, 0.33333334, 0.0,    0.999,      0.999], dtype=np.float32)
        self.expectSequenceEqual(self.env.episode['indicators'][1].tolist(), expected_indicators.tolist(), msg=f"next indicators")

        expected_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.expectSequenceEqual(self.env.episode['state'][1].tolist(), expected_state.tolist(), msg=f"next state")

        expected_norm = 0.0
        self.expectAlmostEqual(self.env.episode['norm'][1], expected_norm, places=5, msg=f"next norm")

        expected_done = 1
        self.expectEqual(self.env.episode['done'][1], expected_done, msg=f"next done")


if __name__ == '__main__':
    unittest.main()
