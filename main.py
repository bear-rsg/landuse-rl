import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import logging
import sys
import random
import numpy as np

from src.dqn import DQN
from src.replaybuffer import ReplayBuffer


class Agent():
    def __init__(self, device, state_size, hidden_size, action_size, replay_memory_size=1e5, batch_size=64, gamma=0.99,
                 learning_rate=1e-3, target_tau=2e-3, update_rate=4):
        self.device = device
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate

        self.network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.learn_rate)  # or optim.SGD or optim.Adam

        # Replay memory
        self.memory = ReplayBuffer(self.device, self.buffer_size, self.batch_size)

        # Initialize time step (for updating every update_rate steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_rate time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, valid_actions, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        # Epsilon-greedy action selection
        r = random.random()
        if r > eps:
            return valid_actions[np.argmax(action_values.cpu().data.numpy()[0][valid_actions])]
        else:
            return random.choice(valid_actions)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get Q values from current observations (s, a) using model nextwork
        Qsa = self.network(states).gather(1, actions)

        #Double DQN
        Qsa_prime_actions = self.network(next_states).detach().max(1)[1].unsqueeze(1)
        Qsa_prime_targets = self.target_network(next_states)[Qsa_prime_actions].unsqueeze(1)

        # Compute Q targets for current states
        Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))

        # Compute loss (error)
        loss = F.huber_loss(Qsa, Qsa_targets)  # or F.mse_loss

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_network, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class Environment():
    def __init__(self, initial_variables, target_indicators):
        self.variables = initial_variables
        self.target = target_indicators
        self.indicators = self.get_indicators()
        self.actions = np.arange(initial_variables.size * 2, dtype=np.int32)  # each variable +-
        self.valid_actions = self.get_valid_actions()
        self.state = self.indicators - self.target
        if np.linalg.norm(self.state) < 1e-5:
            self.done = 1
        else:
            self.done = 0
        self.episode = {'variables': [self.variables],
                        'indicators': [self.indicators],
                        'valid_actions': [self.valid_actions],
                        'state': [self.state],
                        'done': [self.done],
                        'action': [],
                        'reward': []}

    def get_indicators(self):
        indicators = 0.5 * self.variables  # black-box here
        return indicators

    def get_valid_actions(self):
        """
        [var1+, var1-, var2+, var2-, ..., varN+, varN-]
        """
        min_variable_val = 0.0
        max_variable_val = 24.0

        valid_actions = []
        for action in self.actions:
            var_id = action // 2

            if action % 2 == 0:
                if self.variables[var_id] < max_variable_val:
                    valid_actions.append(action)
            elif action % 2 == 1:
                if self.variables[var_id] > min_variable_val:
                    valid_actions.append(action)

        return np.array(valid_actions, dtype=np.int32)

    def get_reward(self):
        prev_state_norm = np.linalg.norm(self.episode['state'][-2])
        curr_state_norm = np.linalg.norm(self.episode['state'][-1])
        reward = prev_state_norm - curr_state_norm

        return reward

    def step(self, action):
        if action not in self.valid_actions:
            logging.error("Illegal move")
            logging.error("Need debugging. Check agent.act() routine")
            raise ValueError(f"{action=} is invalid")

        var_id = action // 2
        if action % 2 == 0:
            self.variables[var_id] += 1  # +1 action of this variable
        elif action % 2 == 1:
            self.variables[var_id] -= 1  # -1 action of this variable

        self.indicators = self.get_indicators()
        self.valid_actions = self.get_valid_actions()
        self.state = self.indicators - self.target
        if np.linalg.norm(self.state) < 1e-5:
            self.done = 1
        else:
            self.done = 0
        self.episode['variables'].append(self.variables)
        self.episode['indicators'].append(self.indicators)
        self.episode['valid_actions'].append(self.valid_actions)
        self.episode['state'].append(self.state)
        self.episode['done'].append(self.done)

        self.episode['action'].append(action)

        reward = self.get_reward()
        self.episode['reward'].append(reward)


def evaluate(agent, max_num_steps_per_episode):
    initial_variables = np.array([1.0, 0.0, 24.0, 13.0, 7.0, 23.0, 24.0, 2.0], dtype=np.float32)
    target_indicators = np.array([2.5, 0.5, 10.5, 7.0, 3.5, 5.0, 11.0, 4.5], dtype=np.float32)
    env = Environment(initial_variables, target_indicators)

    logging.info(f"Target indicators: {target_indicators}")

    # get initial state of the unity environment
    state = env.episode['state'][-1]

    done = env.episode['done'][-1]
    if done:
        logging.info('Convergence criteria met')

    logging.info(f"Current indicators: {env.episode['indicators'][-1]}")

    for i_step in range(1, max_num_steps_per_episode+1):
        valid_actions = env.episode['valid_actions'][-1]

        # determine epsilon-greedy action from current sate
        action = agent.act(state, valid_actions)

        # send the action to the environment
        env.step(action)

        logging.info(f"Current indicators: {env.episode['indicators'][-1]}")

        next_state = env.episode['state'][-1]    # get the next state
        reward = env.episode['reward'][-1]       # get the reward
        done = env.episode['done'][-1]           # see if episode has finished

        # set new state to current state for determining next action
        state = next_state

        if done:
            break

    # logging.info(f"Target indicators: {target_indicators}")


def train(agent, num_episodes, max_num_steps_per_episode, epsilon, epsilon_min, epsilon_decay,
          results, scores_average_window):
    for i_episode in range(1, num_episodes+1):
        # reset the environment
        initial_variables = np.array([1.0, 0.0, 24.0, 13.0, 7.0, 23.0, 24.0, 2.0], dtype=np.float32)
        target_indicators = np.array([2.5, 0.5, 10.5, 7.0, 3.5, 5.0, 11.0, 4.5], dtype=np.float32)
        env = Environment(initial_variables, target_indicators)

        # get initial state of the unity environment
        state = env.episode['state'][-1]

        done = env.episode['done'][-1]
        if done:
            logging.info('Convergence criteria met')
            continue

        # set the initial episode score to zero.
        score = 0

        for i_step in range(1, max_num_steps_per_episode+1):
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
        average_score = np.mean(results['scores'][i_episode-min(i_episode, scores_average_window):i_episode+1])

        # Decrease epsilon for epsilon-greedy policy by decay rate
        # Use max method to make sure epsilon doesn't decrease below epsilon_min
        epsilon = max(epsilon_min, epsilon_decay*epsilon)

        # (Over-) Print current average score
        # logging.info('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

        # Print average score every scores_average_window episodes
        if i_episode % scores_average_window == 0:
            logging.info('Episode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level: -v INFO, -vv DEBUG")
    parser.add_argument('--seed', type=int, help="An integer to be used as seed. If skipped, current time will be used as seed")
    args = parser.parse_args()

    # Configure logging (verbosity level, format etc.)
    args.verbose = 30 - (10 * args.verbose)  # Modify the first number accordingly to enable specific levels by default
    logging.basicConfig(stream=sys.stdout, level=args.verbose, format='%(asctime)s.%(msecs)03d %(levelname)-8s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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

    agent2 = Agent(device, state_size, hidden_size, action_size)  # random agent, untrained
    evaluate(agent, max_num_steps_per_episode)


if __name__ == "__main__":
    main()
