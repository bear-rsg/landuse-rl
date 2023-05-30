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
                 sig_type_interval=(0, 15),
                 sig_type_step=1,
                 land_use_interval=(-1, 1),
                 land_use_n_points=21,
                 greenspace_interval=(0, 1),
                 greenspace_n_points=11,
                 job_type_interval=(0, 1),
                 job_type_n_points=11,
                 epsilon=0.001,
                 max_air_quality=1.0,
                 max_house_price=1.0,
                 max_job_accessibility=1.0,
                 max_greenspace_accessibility=1.0,
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
            self.eng = EngineDebug(self.variables)
        else:
            self.eng = Engine(self.variables, random_seed=seed)

        self.n_indicators = self.eng.indicators.size  # Number of nodes in the input layer

        # Calculate the flattened and normalised indicators based on the current variables
        self.get_indicators()

        # Create the actions array and calculate the valid actions
        self.actions = np.arange(self.variables.size * 2, dtype=np.int32)  # each variable +-
        self.valid_actions = self.get_valid_actions()

    def flatten_normalise_indicators(self, df):
        if df.shape[1] != len(self.max_indicator_values):
            raise IndexError("Check number of columns of the dataframe")

        arr = np.empty(self.n_indicators, dtype=np.float32)
        k = 0
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                arr[k] = df.iloc[i, j] / self.max_indicator_values[j]
                k += 1

        return arr

    def get_indicators(self):
        self.indicators = self.flatten_normalise_indicators(self.eng.indicators)

