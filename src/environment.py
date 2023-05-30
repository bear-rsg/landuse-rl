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
                 seed=1,
                 debug=False):
        # Create the possible values for each variable
        self.sig_type_values = np.arange(sig_type_interval[0], sig_type_interval[1] + 1, sig_type_step, dtype=np.int32)
        self.land_use_values = np.linspace(land_use_interval[0] + epsilon, land_use_interval[1] - epsilon, num=land_use_n_points, endpoint=True, dtype=np.float32)
        self.greenspace_values = np.linspace(greenspace_interval[0], greenspace_interval[1] - epsilon, num=greenspace_n_points, endpoint=True, dtype=np.float32)
        self.job_type_values = np.linspace(job_type_interval[0] + epsilon, job_type_interval[1] - epsilon, num=job_type_n_points, endpoint=True, dtype=np.float32)
        self.variable_values = [self.sig_type_values, self.land_use_values, self.greenspace_values, self.job_type_values]

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

