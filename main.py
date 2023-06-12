import torch
import argparse
import logging
import sys
import platform
import random
import numpy as np
import pandas as pd

from src.agent import Agent
from src.environment import Environment
from src.tasks import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level: -v INFO, -vv DEBUG")
    parser.add_argument('--seed', type=int, help="An integer to be used as seed. If skipped, current time will be used as seed")
    args = parser.parse_args()

    # Configure logging (verbosity level, format etc.)
    args.verbose = 30 - (10 * args.verbose)  # Modify the first number accordingly to enable specific levels by default
    logging.basicConfig(stream=sys.stdout, level=args.verbose, format='%(asctime)s.%(msecs)03d %(levelname)-8s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Print software versions
    logging.debug(f'Python version: {platform.python_version()}')
    logging.debug(f'Numpy version: {np.__version__}')
    logging.debug(f'Pandas version: {pd.__version__}')
    logging.debug(f'Torch version: {torch.__version__}')

    # Set the seed for the random number generators
    if args.seed is not None:
        # if the user provided a seed via the --seed command line argument
        # use it both for torch and random
        logging.info(f"Set the global seed to {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        # If the user didn't provide a seed, we let torch to randomly generate
        # a seed, and we also use the same for the random module.
        random.seed(torch.initial_seed())  # Set random's seed to the same as the one generated for torch
        logging.info(f"Initial seed (both for torch and random) was set to {torch.initial_seed()}")

    # Get cpu or gpu device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using {device} device")

    # # Additional info when using cuda
    # if device.type == 'cuda':
    #     logging.info(torch.cuda.get_device_name(0))

    # Agent parameters
    state_size = 3 * 4                  # n_areas x n_indicators_per_area (719 x 4)
    action_size = 3 * 4 * 2             # n_areas x n_variables_per_area x 2 (719 x 4 x 2)
    hidden_size = action_size
    replay_memory_size = 5000
    batch_size = 32
    gamma = 0.99
    learning_rate = 1e-3
    target_tau = 2e-3
    update_rate = 8

    # Environment parameters
    sig_type_interval = (0, 15)
    sig_type_step = 1
    land_use_interval = (-1, 1)
    land_use_n_points = 21
    greenspace_interval = (0, 1)
    greenspace_n_points = 11
    job_type_interval = (0, 1)
    job_type_n_points = 11
    max_air_quality = 7.5
    max_house_price = 0.5
    max_job_accessibility = 0.5
    max_greenspace_accessibility = 0.5

    # Training parameters
    num_episodes = 100
    max_num_steps_per_episode = 200
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999

    # Initialize the list of targets
    # During training, for every episode we will randomly select one of
    # those as the target indicators.
    target_indicators = []

    data = {'air_quality': np.array([5.5, 4.0, 2.5], dtype=np.float64),
            'house_price': np.array([0.4995, 0.4995, 0.0], dtype=np.float64),
            'job_accessibility': np.array([0.0, 0.4995, 0.4995], dtype=np.float64),
            'greenspace_accessibility': np.array([0.2001, 0.0504, 0.4995], dtype=np.float64)}
    target_indicators.append(pd.DataFrame(data))

    data = {'air_quality': np.array([5.0, 3.5, 2.5], dtype=np.float64),
            'house_price': np.array([0.4995, 0.4995, 0.0], dtype=np.float64),
            'job_accessibility': np.array([0.0, 0.4995, 0.4995], dtype=np.float64),
            'greenspace_accessibility': np.array([0.2001, 0.0504, 0.4995], dtype=np.float64)}
    target_indicators.append(pd.DataFrame(data))

    data = {'air_quality': np.array([5.0, 4.0, 4.0], dtype=np.float64),
            'house_price': np.array([0.4995, 0.4995, 0.0], dtype=np.float64),
            'job_accessibility': np.array([0.0, 0.4995, 0.4995], dtype=np.float64),
            'greenspace_accessibility': np.array([0.2001, 0.0504, 0.4995], dtype=np.float64)}
    target_indicators.append(pd.DataFrame(data))

    data = {'air_quality': np.array([5.0, 4.0, 2.5], dtype=np.float64),
            'house_price': np.array([0.3996, 0.4995, 0.0], dtype=np.float64),
            'job_accessibility': np.array([0.0, 0.4995, 0.4995], dtype=np.float64),
            'greenspace_accessibility': np.array([0.2001, 0.0504, 0.4995], dtype=np.float64)}
    target_indicators.append(pd.DataFrame(data))

    # Initialise the agent
    agent = Agent(device, state_size, hidden_size, action_size, replay_memory_size=replay_memory_size, batch_size=batch_size,
                  gamma=gamma, learning_rate=learning_rate, target_tau=target_tau, update_rate=update_rate)

    # Start the episodes loop to train
    for i_episode in range(1, num_episodes + 1):
        # Reset the environment
        target = random.choice(target_indicators)  # Randomly select a target for this episode
        env = Environment(target,
                          sig_type_interval=sig_type_interval,
                          sig_type_step=sig_type_step,
                          land_use_interval=land_use_interval,
                          land_use_n_points=land_use_n_points,
                          greenspace_interval=greenspace_interval,
                          greenspace_n_points=greenspace_n_points,
                          job_type_interval=job_type_interval,
                          job_type_n_points=job_type_n_points,
                          max_air_quality=max_air_quality,
                          max_house_price=max_house_price,
                          max_job_accessibility=max_job_accessibility,
                          max_greenspace_accessibility=max_greenspace_accessibility,
                          debug=True)

        train(agent, env, max_num_steps_per_episode, epsilon)

        # Decrease epsilon for epsilon-greedy policy by decay rate
        # Use max method to make sure epsilon doesn't decrease below epsilon_min
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Print info about episode
        steps = len(env.episode['reward'])
        score = sum(env.episode['reward'])
        logging.info(f"Episode {i_episode}\tSteps: {steps}\tScore: {score:.2f}")
        logging.debug(f"epsilon: {epsilon}")

        # # Calculate mean score over last 'scores_average_window' episodes
        # # Mean score is calculated over current episodes until i_episode > 'scores_average_window'
        # score = sum(env.episode['reward'])  # The score for the episode is the sum of the rewards
        # results['scores'].append(score)  # Add episode score to scores
        # average_score = np.mean(results['scores'][i_episode - min(i_episode, scores_average_window):i_episode + 1])

        # # (Over-) Print current average score
        # # logging.info('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

        # # Print average score every scores_average_window episodes
        # if i_episode % scores_average_window == 0:
        #     logging.info("Episode {}\tAverage Score: {:.2f}".format(i_episode, average_score))

    # agent2 = Agent(device, state_size, hidden_size, action_size)  # random agent, untrained
    # evaluate(agent, max_num_steps_per_episode)


if __name__ == "__main__":
    main()
