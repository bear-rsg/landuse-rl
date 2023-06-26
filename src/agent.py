import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import pprint
import numpy as np
import logging
from pathlib import Path
from src.dqn import DQN
from src.replaybuffer import ReplayBuffer


from torchsummary import summary

class Agent():

    def __init__(self, device, state_size, hidden_size, action_size, save_all_steps, save_freq, result_dir_path, replay_memory_size=1e5, batch_size=64, gamma=0.99,
                 learning_rate=1e-3, beta1=0.9, beta2=0.999, target_tau=2e-3, update_rate=4):
        self.device = device
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta1=beta1
        self.beta2=beta2
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate

        self.network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.learn_rate,betas=(self.beta1,self.beta2))  

        # Replay memory
        self.memory = ReplayBuffer(self.device, self.buffer_size, self.batch_size)

        # Initialize time step (for updating every update_rate steps)
        self.t_step = 0
        self.printedsummary = False
        self.save_all_steps = save_all_steps
        self.save_freq = save_freq
        self.result_dir_path = result_dir_path
        self.start_episode = 0
        self.save_episode = 0

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


    def learn(self, experiences: tuple[torch.Tensor], gamma: float) -> None:
        """Update value parameters using given batch of experience tuples.

            Args:
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get Q values from current observations (s, a) using model nextwork
        Qsa = self.network(states).gather(dim=1, index=actions) 

        #Double DQN
        Qsa_prime_actions = self.network(next_states).detach().max(1)[1].unsqueeze(dim=1) 
        Qsa_prime_targets = self.target_network(next_states)[Qsa_prime_actions]

        # Compute Q targets for current states
        Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))
        if not self.printedsummary: 
            logging.info("states.shape")
            logging.info(states.shape)
            logging.info("actions.shape")
            logging.info(actions.shape)
            logging.info("rewards.shape")
            logging.info(rewards.shape)
            logging.info("next_states.shape")
            logging.info(next_states.shape)
            logging.info(summary(self.network.model, input_size=(1, states.shape[0], states.shape[1])))
            logging.info("Qsa.shape")
            logging.info(Qsa.shape)
            logging.info("Qsa_targets.shape")
            logging.info(Qsa_targets.shape)
            self.printedsummary = True
            
        # Compute loss (error) 
        #loss = F.huber_loss( Qsa,Qsa_targets)  
        loss = F.huber_loss( Qsa.expand_as(Qsa_targets),Qsa_targets)  # or F.mse_loss
        # loss = F.mse_loss(Qsa, Qsa_targets)

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

            
    def save(self, save_episode):
        if not Path(self.result_dir_path).exists():
            Path(self.result_dir_path).mkdir()

        with (Path(self.result_dir_path) / 'arguments.txt').open('w') as f:
            f.write(pprint.pformat(self.__dict__))
        
        save_dir = self.result_dir_path
        self.save_episode = save_episode
        torch.save({
            'optimizer' : self.optimizer.state_dict(), 
            'network' : self.network.state_dict(), 
            'target_network' : self.target_network.state_dict(), 
            'finish_episode' : self.save_episode,
            'start_episode' : self.start_episode,
            'result_path' : str(save_dir),
            "agent-hyperparameters": { 
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
                "gamma": self.gamma,
                "update_frequency": self.update_rate
            }
            }, str(save_dir+'/agent_{}_episode.pkl'.format(save_episode)))

        
        logging.info("============= save success =============")
        logging.info("episode from {} to {}".format(self.start_episode, save_episode))
        logging.info("save result path is {}".format(str(self.result_dir_path)))
    
    def load_test(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))  
        self.network.load_state_dict(checkpoint['network'])  

    def load(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.optimizer.load_state_dict(checkpoint['optimizer'])  
        self.network.load_state_dict(checkpoint['network'])  
        self.target_network.load_state_dict(checkpoint['target_network'])  
        self.start_episode = checkpoint['finish_epoch'] + 1
        self.result_path = checkpoint['result_path']

        self.finish_episode = self.episode + self.start_episode - 1

        logging.info("============= load success =============")
        logging.info("episode, start from {} to {}".format(self.start_episode, self.finish_episode))
        logging.info("previous result path is {}".format(checkpoint['result_path']))

