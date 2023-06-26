import torch
import argparse
import logging
import sys
import platform
import random
import numpy as np
import pandas as pd
import os
import pprint
from pathlib import Path
from src.agent import Agent
from src.toyenvironment import ToyEnvironment as Environment
from src.tasks import train


DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')

def parse_args():
    desc = "Landuse Demonstrator RL "
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level: -v INFO, -vv DEBUG")
    parser.add_argument('--seed', type=int, help="An integer to be used as seed. If skipped, current time will be used as seed")
    #parser.add_argument('--cpu', action='store_true', help='If set, use cpu only')
    parser.add_argument('--save_freq', type=int, default=0, help='Save network dump by every `save_freq` episode. if set to 0, save the last result only')
    parser.add_argument('--max_episode', type=int, default=100, help='The  number of episodes to run')
    parser.add_argument('--max_steps_episode', type=int, default=300, help='The max num steps per episode to run')
    parser.add_argument('--batch_size', type=int, default=32, help='Total batch size')
    parser.add_argument('--data_dir', default=DATASET_PATH, help='Path to the train/test data root directory')
    parser.add_argument('--result_dir', type=str, default='./results', help='Path to save generated images and network dump')
    parser.add_argument('--load', type=str, default="", help='Path to load network weights (if non-empty)')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for system') 
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam optimizer parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam optimizer parameter')
    parser.add_argument('--save_all_ep', type=int, default=0, help='If nonzero, save RL network dump by every episode after this episode')

    args = parser.parse_args()
    validate_args(args)

    return args

def validate_args(args):
    print('validating arguments...')
    pprint.pprint(args.__dict__)

    assert args.max_episode >= 1, 'number of maxepisode must be larger than or equal to one'
    assert args.max_steps_episode >= 1, 'number of maxsteps_episode must be larger than or equal to one'
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'
    
    #if args.load != '':
    #    assert os.path.exists(args.load), 'cannot find network dump file'
    #assert os.path.exists(args.pretrain_dump), 'cannot find pretrained network dump file'
    #assert os.path.exists(args.tag_dump), 'cannot find tag metadata pickle file'

    data_dir_path = Path(args.data_dir)
    assert data_dir_path.exists(), 'cannot find data root directory'

    result_dir_path = Path(args.result_dir)
    if not result_dir_path.exists():
        result_dir_path.mkdir()

def main():
    args = parse_args()
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
    device =  torch.device("cpu")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda") 
        logging.info(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        
    logging.info(f"Using {device} device") 
    
    num_episodes = args.max_episode
    max_num_steps_per_episode = args.max_steps_episode
    learning_rate=args.lr
    batch_size= args.batch_size
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    results = {'scores': []}
    scores_average_window = 10

    state_size = 8
    action_size = state_size * 2
    hidden_size = action_size

    # Initialise the agent
    agent = Agent(device, state_size, hidden_size, action_size,args.save_all_ep, args.save_freq, args.result_dir, replay_memory_size=3000, batch_size=batch_size,
                  gamma=0.99, learning_rate=learning_rate, beta1=args.beta1, beta2=args.beta2, target_tau=4e-2, update_rate=8)

    
    for i_episode in range(1, num_episodes + 1):
        # Reset the environment
        initial_variables = np.array([1.0, 0.0, 24.0, 13.0, 7.0, 23.0, 24.0, 2.0], dtype=np.float32)
        target_indicators = np.array([2.5, 0.5, 10.5, 7.0, 3.5, 5.0, 11.0, 4.5], dtype=np.float32)
        env = Environment(initial_variables, target_indicators)

        train(agent, env, max_num_steps_per_episode, epsilon)

        # Decrease epsilon for epsilon-greedy policy by decay rate
        # Use max method to make sure epsilon doesn't decrease below epsilon_min
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Calculate mean score over last 'scores_average_window' episodes
        # Mean score is calculated over current episodes until i_episode > 'scores_average_window'
        score = sum(env.episode['reward'])  # The score for the episode is the sum of the rewards
        results['scores'].append(score)  # Add episode score to scores
        average_score = np.mean(results['scores'][i_episode - min(i_episode, scores_average_window):i_episode + 1])

        # (Over-) Print current average score
        # logging.info('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

        # Print average score every scores_average_window episodes
        if i_episode % scores_average_window == 0:
            logging.info("Episode {}\tAverage Score: {:.2f}".format(i_episode, average_score))

        if i_episode >= agent.save_all_steps > 0:
            agent.save(i_episode)
        elif agent.save_freq > 0 and i_episode % agent.save_freq == 0:
            agent.save(i_episode)

    print("finished... save model training results")

    if agent.save_freq == 0:
        if agent.save_all_steps <= 0:
            agent.save(i_episode+1)
    
    agent.load_test(agent.result_dir_path+'/agent_25_episode.pkl')


if __name__ == "__main__":
    main()
