import numpy as np
from mjrl.algos.model_accel.sampling import generate_paths, generate_perturbed_actions, trajectory_rollout


class MPCPolicy(object):
    def __init__(self, env,
                 plan_horizon,
                 plan_paths=10,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 seed=123,
                 warmstart=True,
                 fitted_model=None,
                 omega=5.0,
                 **kwargs,
                 ):

        # initialize
        self.env, self.seed = env, seed
        self.n, self.m = env.observation_dim, env.action_dim
        self.plan_horizon, self.num_traj = plan_horizon, plan_paths

        if fitted_model is None:
            print("Policy requires a fitted dynamics model")
            quit()
        else:
            self.fitted_model = fitted_model

        # initialize other params
        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]
        self.act_sequence = np.ones((self.plan_horizon, self.m)) * self.mean
        self.init_act_sequence = self.act_sequence.copy()
        self.warmstart = warmstart
        self.omega = omega

    def get_action(self, obs):
        # generate paths
        if type(self.fitted_model) == list:

            # Ensemble case
            # Collect trajectories from different models with same action sequences
            base_act = self.act_sequence
            act_list = [generate_perturbed_actions(base_act, self.filter_coefs)
                        for _ in range(self.num_traj)]
            actions = np.array(act_list)
            paths_list = []
            for model in self.fitted_model:
                paths = trajectory_rollout(actions, model, obs)
                self.env.env.env.compute_path_rewards(paths)
                paths_list.append(paths)
            # consolidate paths
            paths = dict()
            for k in paths_list[0].keys():
                v = np.vstack([p[k] for p in paths_list])
                paths[k] = v
            R = self.score_trajectory_ensemble(paths, paths_list)

        else:
            paths = generate_paths(num_traj=self.num_traj, fitted_model=self.fitted_model,
                                   start_state=obs, base_act=self.act_sequence, filter_coefs=self.filter_coefs)
            self.env.env.env.compute_path_rewards(paths)  # will populate path['rewards']
            R = self.score_trajectory(paths)

        S = np.exp(self.kappa * (R - np.max(R)))
        act = paths["actions"]

        weighted_seq = S * act.T
        act_sequence = np.sum(weighted_seq.T, axis=0) / (np.sum(S) + 1e-6)
        action = act_sequence[0].copy()

        # get updated action sequence
        if self.warmstart:
            self.act_sequence[:-1] = act_sequence[1:]
            self.act_sequence[-1] = self.mean.copy()
        else:
            self.act_sequence = self.init_act_sequence.copy()
        return action

    def score_trajectory_ensemble(self, paths, paths_list):
        num_traj = self.num_traj
        num_models = len(paths_list)
        total_traj = paths['rewards'].shape[0]
        horizon = paths['rewards'].shape[1]
        predictions = [p['observations'] for p in paths_list]
        disagreement = np.std(predictions, axis=0)      # (num_traj, horizon, state_dim)
        disagreement = np.sum(disagreement, axis=(1,2)) # (num_traj,)
        scores = np.zeros(total_traj)
        for i in range(total_traj):
            disagreement_score = disagreement[i // self.num_traj]
            scores[i] = self.omega * disagreement_score
            for t in range(horizon):
                scores[i] += (self.gamma ** t) * paths["rewards"][i][t]
        return scores

    def score_trajectory(self, paths):
        # rewards shape: (num_traj, horizon)
        num_traj = paths["rewards"].shape[0]
        horizon = paths["rewards"].shape[1]
        scores = np.zeros(num_traj)
        for i in range(num_traj):
            scores[i] = 0.0
            for t in range(horizon):
                scores[i] += (self.gamma**t)*paths["rewards"][i][t]
        return scores
