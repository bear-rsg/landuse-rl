import numpy as np
import pandas as pd
from demoland_engine import Engine, get_lsoa_baseline


# Find closest number in a list
def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


def get_lsoa_baseline_debug():
    # initialize values for debugging and unit testing
    data = {'signature_type': np.array([10, 8, 5], dtype=np.int64),
            'use': np.array([-1, 1, 0], dtype=np.int64),
            'greenspace': np.array([0, 1, 1], dtype=np.int64),
            'job_types': np.array([0.45, 0.12, 0.97], dtype=np.float64)}

    # Create the pandas DataFrame
    df = pd.DataFrame(data)

    return df


class EngineDebug():
    def __init__(self, lsoa_input):
        self.variables = lsoa_input
        self.calc_indicators()

    def calc_indicators(self):
        x = lambda a: abs(a) / 2.0  # toy black-box for debugging

        air_quality = np.array([x(a) for a in self.variables['signature_type']], dtype=np.float64)
        house_price = np.array([x(a) for a in self.variables['use']], dtype=np.float64)
        job_accessibility = np.array([x(a) for a in self.variables['greenspace']], dtype=np.float64)
        greenspace_accessibility = np.array([x(a) for a in self.variables['job_types']], dtype=np.float64)

        data = {'air_quality': air_quality,
                'house_price': house_price,
                'job_accessibility': job_accessibility,
                'greenspace_accessibility': greenspace_accessibility}

        self.indicators = pd.DataFrame(data)

    def change(self, idx, val):
        self.variables.iloc[idx[0], idx[1]] = val
        self.calc_indicators()


class Environment():
    def __init__(self,
                 target_indicators,
                 sig_type_interval=(0, 15),
                 sig_type_step=1,
                 land_use_interval=(-1, 1),
                 land_use_n_points=21,
                 greenspace_interval=(0, 1),
                 greenspace_n_points=11,
                 job_type_interval=(0, 1),
                 job_type_n_points=11,
                 epsilon=0.001,
                 max_air_quality=28.0,
                 max_house_price=9.2,
                 max_job_accessibility=300000.0,
                 max_greenspace_accessibility=18500000.0,
                 tolerance=1e-5,
                 seed=1,
                 debug=False):
        # Create the possible values for each variable
        self.sig_type_values = np.arange(sig_type_interval[0], sig_type_interval[1] + 1, sig_type_step, dtype=np.int32)
        self.land_use_values = np.linspace(land_use_interval[0] + epsilon, land_use_interval[1] - epsilon, num=land_use_n_points, endpoint=True, dtype=np.float32)
        self.greenspace_values = np.linspace(greenspace_interval[0], greenspace_interval[1] - epsilon, num=greenspace_n_points, endpoint=True, dtype=np.float32)
        self.job_type_values = np.linspace(job_type_interval[0] + epsilon, job_type_interval[1] - epsilon, num=job_type_n_points, endpoint=True, dtype=np.float32)
        self.variable_values = [self.sig_type_values, self.land_use_values, self.greenspace_values, self.job_type_values]

        self.max_air_quality = max_air_quality
        self.max_house_price = max_house_price
        self.max_job_accessibility = max_job_accessibility
        self.max_greenspace_accessibility = max_greenspace_accessibility
        self.max_indicator_values = [self.max_air_quality, self.max_house_price, self.max_job_accessibility, self.max_greenspace_accessibility]

        self.tolerance = tolerance
        self.debug = debug

        # Load the initial variables
        if self.debug:
            self.variables = get_lsoa_baseline_debug()
        else:
            self.variables = get_lsoa_baseline()  # current state of Newcastle

        # Snap the variables to the possible values
        for j in range(self.variables.shape[1]):
            for i in range(self.variables.shape[0]):
                self.variables.iloc[i, j] = closest(self.variable_values[j], self.variables.iloc[i, j])

        # Initialize the engine
        if self.debug:
            self.eng = EngineDebug(self.variables.copy(deep=True))
        else:
            self.eng = Engine(self.variables.copy(deep=True), random_seed=seed)

        self.n_indicators = self.eng.indicators.size  # Number of nodes in the input layer

        # Calculate the flattened and normalised indicators based on the current variables
        self.get_indicators()

        self.target = self.flatten_normalise_indicators(target_indicators)
        self.state = self.indicators - self.target
        self.norm = np.linalg.norm(self.state)
        self.done = 1 if self.norm < self.tolerance else 0

        # Create the actions array and calculate the valid actions
        self.actions = np.arange(self.variables.size * 2, dtype=np.int32)  # each variable +-
        self.valid_actions = self.get_valid_actions()

        self.episode = {'variables': [self.variables],
                        'indicators': [self.indicators],
                        'valid_actions': [self.valid_actions],
                        'state': [self.state],
                        'done': [self.done],
                        'norm': [self.norm],
                        'action': [],
                        'reward': []}

    def flatten_normalise_indicators(self, df):
        """
        Convert a pandas dataframe into a flattened numpy array
        in a row-wise fashion, i.e. if the dataframe is
        [1 2 3
         4 5 6]
        then the flattened array will be [1, 2, 3, 4, 5, 6].
        We also normalise each column by dividing with its
        corresponding normalisation value.
        """
        if df.shape[1] != len(self.max_indicator_values):
            raise IndexError("Check number of columns of the dataframe")

        arr = np.empty(self.n_indicators, dtype=np.float32)
        k = 0
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                arr[k] = df.iloc[i, j] / self.max_indicator_values[j]
                k += 1

        return arr

    def revert_flatten_normalise_indicators(self, indicators):
        """
        From a flattened and normalised numpy array of indicators
        go back to a 2-dimensional and not normalised pandas dataframe.
        """
        air_quality = indicators[0::4] * self.max_air_quality
        house_price = indicators[1::4] * self.max_house_price
        job_accessibility = indicators[2::4] * self.max_job_accessibility
        greenspace_accessibility = indicators[3::4] * self.max_greenspace_accessibility

        data = {'air_quality': air_quality,
                'house_price': house_price,
                'job_accessibility': job_accessibility,
                'greenspace_accessibility': greenspace_accessibility}

        return pd.DataFrame(data)

    def get_indicators(self):
        self.indicators = self.flatten_normalise_indicators(self.eng.indicators)

    def get_valid_actions(self):
        """
        Select the actions from the following list that if performed will not end up with
        the corresponding variable out of bounds.
        [sig1+, sig1-, use1+, use1-, green1+, green1-, job1+, job1-,
         sig2+, sig2-, use2+, use2-, green2+, green2-, job2+, job2-, ... ]
        """    
        valid_actions = []
        for action in self.actions:
            var_id = action // 2
            var_row_id = var_id // 4
            var_col_id = var_id % 4

            if action % 2 == 0:
                if self.variables.iloc[var_row_id, var_col_id] < self.variable_values[var_col_id][-1]:
                    valid_actions.append(action)
            elif action % 2 == 1:
                if self.variables.iloc[var_row_id, var_col_id] > self.variable_values[var_col_id][0]:
                    valid_actions.append(action)

        return np.array(valid_actions, dtype=np.int32)

    def get_reward(self):
        prev_state_norm = self.episode['norm'][-2]
        curr_state_norm = self.episode['norm'][-1]
        reward = prev_state_norm - curr_state_norm

        return reward

    def step(self, action):
        if action not in self.valid_actions:
            raise ValueError(f"{action=} is invalid")

        change, var_row_id, var_col_id = self.action_id_1d_to_2d(action)
        self.new_variables = self.variables.copy(deep=True)
        if change == 0:  # Increase the variable with index (var_row_id, var_col_id)
            # the smallest element of variable_values greater than the current variable
            next_variable_value = self.variable_values[var_col_id][self.variable_values[var_col_id] > self.variables.iloc[var_row_id, var_col_id]].min()
            self.new_variables.iloc[var_row_id, var_col_id] = next_variable_value
            self.eng.change((var_row_id, var_col_id), next_variable_value)  # Update the engine
        elif change == 1:  # Decrease the variable with index (var_row_id, var_col_id)
            # the largest element of variable_values less than the current variable
            prev_variable_value = self.variable_values[var_col_id][self.variable_values[var_col_id] < self.variables.iloc[var_row_id, var_col_id]].max()
            self.new_variables.iloc[var_row_id, var_col_id] = prev_variable_value
            self.eng.change((var_row_id, var_col_id), prev_variable_value)  # Update the engine
        self.variables = self.new_variables

        self.get_indicators()
        self.state = self.indicators - self.target
        self.norm = np.linalg.norm(self.state)
        self.done = 1 if self.norm < self.tolerance else 0
        self.valid_actions = self.get_valid_actions()

        self.episode['variables'].append(self.variables)
        self.episode['indicators'].append(self.indicators)
        self.episode['valid_actions'].append(self.valid_actions)
        self.episode['state'].append(self.state)
        self.episode['done'].append(self.done)
        self.episode['norm'].append(self.norm)
        self.episode['action'].append(action)
        self.episode['reward'].append(self.get_reward())

    def action_id_1d_to_2d(self, action):
        """
        Input:
            action: integer
        Returns:
            change: integer (0 for increase and 1 for decrease)
            var_row_id: integer (row id of the variable to change)
            var_col_id: integer (col id of the variable to change)
        """
        change = action % 2
        var_id = action // 2
        var_row_id = var_id // 4
        var_col_id = var_id % 4

        return change, var_row_id, var_col_id
