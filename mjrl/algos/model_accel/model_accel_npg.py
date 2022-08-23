import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
import mjrl.envs
import os
import time as timer
from torch.autograd import Variable
from mjrl.utils.gym_env import GymEnv
from mjrl.algos.model_accel.nn_dynamics import WorldModel
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.algos.model_accel.sampling import policy_rollout

# Import NPG
from mjrl.algos.npg_cg import NPG


class ModelAccelNPG(NPG):
    def __init__(self, learned_model=None,
                 refine=False,
                 kappa=5.0,
                 plan_horizon=10,
                 plan_paths=100,
                 reward_function=None,
                 termination_function=None,
                 **kwargs):
        super(ModelAccelNPG, self).__init__(**kwargs)
        if learned_model is None:
            print("Algorithm requires a (list of) learned dynamics model")
            quit()
        elif isinstance(learned_model, WorldModel):
            self.learned_model = [learned_model]
        else:
            self.learned_model = learned_model
        self.refine, self.kappa, self.plan_horizon, self.plan_paths = refine, kappa, plan_horizon, plan_paths
        self.reward_function, self.termination_function = reward_function, termination_function

    def to(self, device):
        # Convert all the networks (except policy network which is clamped to CPU)
        # to the specified device
        for model in self.learned_model:
            model.to(device)
        try:    self.baseline.model.to(device)
        except: pass

    def is_cuda(self):
        # Check if any of the networks are on GPU
        model_cuda = [model.is_cuda() for model in self.learned_model]
        model_cuda = any(model_cuda)
        baseline_cuda = next(self.baseline.model.parameters()).is_cuda
        return any([model_cuda, baseline_cuda])

    def train_step(self, N,
                   env=None,
                   sample_mode='trajectories',
                   horizon=1e6,
                   gamma=0.995,
                   gae_lambda=0.97,
                   num_cpu='max',
                   env_kwargs=None,
                   init_states=None,
                   reward_function=None,
                   termination_function=None,
                   truncate_lim=None,
                   truncate_reward=0.0,
                   **kwargs,
                   ):

        ts = timer.time()

        # get the correct env behavior
        if env is None:
            env = self.env
        elif type(env) == str:
            env = GymEnv(env)
        elif isinstance(env, GymEnv):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError

        # get correct behavior for reward and termination
        reward_function = self.reward_function if reward_function is None else reward_function
        termination_function = self.termination_function if termination_function is None else termination_function
        if reward_function: assert callable(reward_function)
        if termination_function: assert callable(termination_function)

        # simulate trajectories with the learned model(s)
        # we want to use the same task instances (e.g. goal locations) for each model in ensemble
        paths = []

        # NOTE: We can optionally specify a set of initial states to perform the rollouts from
        # This is useful for starting rollouts from the states in the replay buffer
        init_states = np.array([env.reset() for _ in range(N)]) if init_states is None else init_states
        assert type(init_states) == list
        assert len(init_states) == N

        for model in self.learned_model:
            # dont set seed explicitly -- this will make rollouts follow tne global seed
            rollouts = policy_rollout(num_traj=N, env=env, policy=self.policy,
                                      learned_model=model, eval_mode=False, horizon=horizon,
                                      init_state=init_states, seed=None)
            # use learned reward function if available
            if model.learn_reward:
                model.compute_path_rewards(rollouts)
            else:
               rollouts = reward_function(rollouts)
            num_traj, horizon, state_dim = rollouts['observations'].shape
            for i in range(num_traj):
                path = dict()
                obs = rollouts['observations'][i, :, :]
                act = rollouts['actions'][i, :, :]
                rew = rollouts['rewards'][i, :]
                path['observations'] = obs
                path['actions'] = act
                path['rewards'] = rew
                path['terminated'] = False
                paths.append(path)

        # NOTE: If tasks have termination condition, we will assume that the env has
        # a function that can terminate paths appropriately.
        # Otherwise, termination is not considered.

        if callable(termination_function): paths = termination_function(paths)

        # remove paths that are too short
        paths = [path for path in paths if path['observations'].shape[0] >= 5]

        # additional truncation based on error in the ensembles
        if truncate_lim is not None and len(self.learned_model) > 1:
            for path in paths:
                pred_err = np.zeros(path['observations'].shape[0] - 1)
                for model in self.learned_model:
                    s = path['observations'][:-1]
                    a = path['actions'][:-1]
                    s_next = path['observations'][1:]
                    pred = model.predict(s, a)
                    model_err = np.mean((s_next - pred)**2, axis=-1)
                    pred_err = np.maximum(pred_err, model_err)
                violations = np.where(pred_err > truncate_lim)[0]
                truncated = (not len(violations) == 0)
                T = violations[0] + 1 if truncated else obs.shape[0]
                T = max(4, T)   # we don't want corner cases of very short truncation
                path["observations"] = path["observations"][:T]
                path["actions"] = path["actions"][:T]
                path["rewards"] = path["rewards"][:T]
                if truncated: path["rewards"][-1] += truncate_reward
                path["terminated"] = False if T == obs.shape[0] else True

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # log number of samples
        if self.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            self.logger.log_kv('num_samples', num_samples)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

    def get_action(self, observation):
        if self.refine is False:
            return self.policy.get_action(observation)
        else:
            return self.get_refined_action(observation)

    def get_refined_action(self, observation):
        # TODO(Aravind): Implemenet this
        # This function should rollout many trajectories according to the learned
        # dynamics model and the policy, and should refine around the policy by
        # incorporating reward based refinement
        raise NotImplementedError
