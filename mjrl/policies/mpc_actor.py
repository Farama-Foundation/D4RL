import numpy as np
from trajopt.utils import gather_paths_parallel


class MPCActor(object):
    def __init__(self, env, H, paths_per_cpu,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 seed=123,
                 ):

        self.env, self.seed = env, seed
        self.n, self.m = env.observation_dim, env.action_dim
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu

        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]

        self.env.reset()
        self.env.set_seed(seed)
        self.env.reset(seed=seed)
        self.act_sequence = np.ones((self.H, self.m)) * self.mean
        self.ctr = 1

    def score_trajectory(self, paths):
        scores = np.zeros(len(paths))
        for i in range(len(paths)):
            scores[i] = 0.0
            for t in range(paths[i]["rewards"].shape[0]):
                scores[i] += (self.gamma**t)*paths[i]["rewards"][t]
        return scores

    def get_action(self, env_state):
        # Set to env_state
        # Shoot trajectories
        # Return optimal action
        seed = self.seed + self.ctr * 1000
        paths = gather_paths_parallel(self.env.env_id,
                                      env_state,
                                      self.act_sequence,
                                      self.filter_coefs,
                                      seed,
                                      self.paths_per_cpu,
                                      self.num_cpu,
                                      )

        num_traj = len(paths)
        R = self.score_trajectory(paths)
        S = np.exp(self.kappa*(R-np.max(R)))
        act = np.sum([paths[i]["actions"][0] * S[i] for i in range(num_traj)], axis=0)
        act = act / (np.sum(S) + 1e-6)
        return act