import numpy as np
import copy

class ZeroBaseline:
    def __init__(self, env_spec, **kwargs):
        n = env_spec.observation_dim       # number of states
        self._coeffs = None

    def fit(self, paths, return_errors=False):
        if return_errors:
            return 1.0, 1.0

    def predict(self, path):
        return np.zeros(len(path["rewards"]))
