import numpy as np


class ToyEnvironment():
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
            raise ValueError(f"{action=} is invalid (Check agent.act() routine)")

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
