import torch
import torch.optim as optim
import torch.nn.functional as F
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
