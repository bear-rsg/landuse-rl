import unittest
import numpy as np
import pandas as pd
import random
import torch

from tests.mytestcase import MyTestCase
from src.agent import Agent
from src.toyenvironment import ToyEnvironment


def train(agent, num_episodes, max_num_steps_per_episode, epsilon, epsilon_min, epsilon_decay,
          results):
    for i_episode in range(1, num_episodes + 1):
        # reset the environment
        initial_variables = np.array([1.0, 0.0, 24.0, 13.0, 7.0, 23.0, 24.0, 2.0], dtype=np.float32)
        target_indicators = np.array([2.5, 0.5, 10.5, 7.0, 3.5, 5.0, 11.0, 4.5], dtype=np.float32)
        env = ToyEnvironment(initial_variables, target_indicators)

        # get initial state of the unity environment
        state = env.episode['state'][-1]

        done = env.episode['done'][-1]
        if done:
            continue

        # set the initial episode score to zero.
        score = 0

        for i_step in range(1, max_num_steps_per_episode + 1):
            valid_actions = env.episode['valid_actions'][-1]

            # determine epsilon-greedy action from current sate
            action = agent.act(state, valid_actions, epsilon)

            # send the action to the environment
            env.step(action)

            next_state = env.episode['state'][-1]    # get the next state
            reward = env.episode['reward'][-1]       # get the reward
            done = env.episode['done'][-1]           # see if episode has finished

            #Send (S, A, R, S') info to the DQN agent for a neural network update
            agent.step(state, action, reward, next_state, done)

            # set new state to current state for determining next action
            state = next_state

            # Update episode score
            score += reward

            # If this episode is done,
            # then exit episode loop, to begin new episode
            if done:
                break

        # Add episode score to Scores and...
        # Calculate mean score over last 'scores_average_window' episodes
        # Mean score is calculated over current episodes until i_episode > 'scores_average_window'
        results['scores'].append(score)

        # Decrease epsilon for epsilon-greedy policy by decay rate
        # Use max method to make sure epsilon doesn't decrease below epsilon_min
        epsilon = max(epsilon_min, epsilon_decay * epsilon)


class TestIntegrationToyEnv(MyTestCase):
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

        state_size = 8
        action_size = state_size * 2
        hidden_size = action_size

        agent = Agent(device, state_size, hidden_size, action_size, replay_memory_size=3000, batch_size=32,
                      gamma=0.99, learning_rate=1e-2, target_tau=4e-2, update_rate=8)

        train(agent, num_episodes, max_num_steps_per_episode, epsilon, epsilon_min, epsilon_decay,
              results)

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

        state_size = 8
        action_size = state_size * 2
        hidden_size = action_size

        agent = Agent(device, state_size, hidden_size, action_size, replay_memory_size=3000, batch_size=32,
                      gamma=0.99, learning_rate=1e-2, target_tau=4e-2, update_rate=8)

        train(agent, num_episodes, max_num_steps_per_episode, epsilon, epsilon_min, epsilon_decay,
              results)

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
