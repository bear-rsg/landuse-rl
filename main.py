import torch
import argparse
import logging
import sys
import platform
import random
import numpy as np
import pandas as pd

from src.agent import Agent
from src.toyenvironment import ToyEnvironment as Environment
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
    #     logging.info('Memory Usage:')
    #     logging.info('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    #     logging.info('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    num_episodes = 100
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

    # agent2 = Agent(device, state_size, hidden_size, action_size)  # random agent, untrained
    # evaluate(agent, max_num_steps_per_episode)


if __name__ == "__main__":
    main()
