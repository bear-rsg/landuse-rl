import torch
import torch.nn as nn
from collections import OrderedDict

class DQN(torch.nn.Module):

    def __init__(self, state_size, hidden_size, action_size):
        # weights and bias are initialized from uniform(−sqrt(k),sqrt(k)), where k=1/in_features.
        # This is similar, but not same, to Kaiming (He) uniform initialization.
        super(DQN, self).__init__()

        self.model = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(state_size, hidden_size)),   # input  -> hidden
            ('relu1', torch.nn.LeakyReLU(negative_slope=0.01)),  # or ReLU
            ('fc2', torch.nn.Linear(hidden_size, action_size)),  # hidden -> output
        ]))
        
    def forward(self, state):
        return self.model(state)
