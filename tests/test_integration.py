import unittest
import numpy as np
import pandas as pd
import random
import torch

from tests.mytestcase import MyTestCase
from main import Agent, train


class TestIntegration(MyTestCase):
    def test_integration_toy_env_1(self):
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        num_episodes = 30
        max_num_steps_per_episode = 300
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay = 0.995
        results = {'scores': []}
        scores_average_window = 10

        state_size = 8
        action_size = state_size * 2
        hidden_size = action_size

        agent = Agent(device, state_size, hidden_size, action_size, replay_memory_size=3000, batch_size=32,
                      gamma=0.99, learning_rate=1e-2, target_tau=4e-2, update_rate=8)

        train(agent, num_episodes, max_num_steps_per_episode, epsilon, epsilon_min, epsilon_decay,
              results, scores_average_window)

        expected = np.array([0.192464, -0.521434, -1.714819, -0.156871, 3.559739,
                             2.080391,  3.226177,  1.994953,  3.921239, 1.467250,
                             0.592607,  0.696119,  3.226177, -1.040418, 2.123585,
                             1.163054,  1.745763,  5.291790,  3.858737, 2.766391,
                             5.598578,  1.869069,  4.688304,  6.231554, 4.767367,
                             3.736555,  4.389866,  5.710417,  5.953375, 5.197284], dtype=np.float32)
        for i in range(expected.shape[0]):
            self.expectAlmostEqual(results['scores'][i], expected[i], places=5, msg=f"{i=}")

    def test_integration_toy_env_2(self):
        seed = 1
        random.seed(seed)
        torch.manual_seed(seed)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        num_episodes = 30
        max_num_steps_per_episode = 300
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay = 0.995
        results = {'scores': []}
        scores_average_window = 10

        state_size = 8
        action_size = state_size * 2
        hidden_size = action_size

        agent = Agent(device, state_size, hidden_size, action_size, replay_memory_size=3000, batch_size=32,
                      gamma=0.99, learning_rate=1e-2, target_tau=4e-2, update_rate=8)

        train(agent, num_episodes, max_num_steps_per_episode, epsilon, epsilon_min, epsilon_decay,
              results, scores_average_window)

        expected = np.array([1.994953,  4.462039, 0.290524, -1.261635, 3.334650,
                             2.080391, -1.234276, 0.524406,  1.275488, 2.914929,
                             2.622039,  0.943645, 2.527964,  4.049293, 3.736555,
                             2.037517,  1.016003, 0.524406,  4.114949, 1.869069,
                             4.114949,  3.676792, 4.535764,  3.797190, 4.462039,
                             4.931826,  4.848485, 5.491950,  4.462039, 5.491950], dtype=np.float32)
        for i in range(expected.shape[0]):
            self.expectAlmostEqual(results['scores'][i], expected[i], places=5, msg=f"{i=}")


if __name__ == '__main__':
    unittest.main()
