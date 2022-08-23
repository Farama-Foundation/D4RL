import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve
from mjrl.algos.batch_reinforce import BatchREINFORCE


class PPO(BatchREINFORCE):
    def __init__(self, env, policy, baseline,
                 clip_coef = 0.2,
                 epochs = 10,
                 mb_size = 64,
                 learn_rate = 3e-4,
                 seed = 123,
                 save_logs = False,
                 **kwargs
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.learn_rate = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.clip_coef = clip_coef
        self.epochs = epochs
        self.mb_size = mb_size
        self.running_score = None
        if save_logs: self.logger = DataLog()

        self.optimizer = torch.optim.Adam(self.policy.trainable_params, lr=learn_rate)

    def PPO_surrogate(self, observations, actions, advantages):
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        LR_clip = torch.clamp(LR, min=1-self.clip_coef, max=1+self.clip_coef)
        ppo_surr = torch.mean(torch.min(LR*adv_var,LR_clip*adv_var))
        return ppo_surr

    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        # NOTE : advantage should be zero mean in expectation
        # normalized step size invariant to advantage scaling,
        # but scaling can help with least squares

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        params_before_opt = self.policy.get_param_values()

        ts = timer.time()
        num_samples = observations.shape[0]
        for ep in range(self.epochs):
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                obs = observations[rand_idx]
                act = actions[rand_idx]
                adv = advantages[rand_idx]
                self.optimizer.zero_grad()
                loss = - self.PPO_surrogate(obs, act, adv)
                loss.backward()
                self.optimizer.step()

        params_after_opt = self.policy.get_param_values()
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        t_opt = timer.time() - ts

        # Log information
        if self.save_logs:
            self.logger.log_kv('t_opt', t_opt)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

        return base_stats
