"""
Minimize bc loss (MLE, MSE, RWR etc.) with pytorch optimizers
"""

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
from tqdm import tqdm


class BC:
    def __init__(self, expert_paths,
                 policy,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 optimizer = None,
                 loss_type = 'MSE',  # can be 'MLE' or 'MSE'
                 save_logs = True,
                 set_transforms = False,
                 log_dir="./logs",
                 **kwargs,
                 ):

        self.policy = policy
        self.expert_paths = expert_paths
        self.epochs = epochs
        self.mb_size = batch_size
        self.logger = DataLog(log_dir)
        self.loss_type = loss_type
        self.save_logs = save_logs
        self.eval_every = kwargs["eval_every"]
        self.eval_func = kwargs["eval_func"]
        self.device = kwargs["device"]

        if set_transforms:
            in_shift, in_scale, out_shift, out_scale = self.compute_transformations()
            self.set_transformations(in_shift, in_scale, out_shift, out_scale)
            self.set_variance_with_data(out_scale)

        # construct optimizer
        self.optimizer = torch.optim.Adam(list(self.policy.model.parameters()) + [self.policy.log_std], lr=lr) if optimizer is None else optimizer

        # Loss criterion if required
        if loss_type == 'MSE':
            self.loss_criterion = torch.nn.MSELoss()

    def compute_transformations(self):
        # get transformations
        if self.expert_paths == [] or self.expert_paths is None:
            in_shift, in_scale, out_shift, out_scale = None, None, None, None
        else:
            observations = np.concatenate([path["observations"] for path in self.expert_paths])
            actions = np.concatenate([path["actions"] for path in self.expert_paths])
            in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)
        return in_shift, in_scale, out_shift, out_scale

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # set scalings in the target policy
        self.policy.model.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.policy.old_model.set_transformations(in_shift, in_scale, out_shift, out_scale)

    def set_variance_with_data(self, out_scale):
        # set the variance of gaussian policy based on out_scale
        params = self.policy.get_param_values()
        params[-self.policy.m:] = np.log(out_scale + 1e-12)
        self.policy.set_param_values(params)

    def loss(self, data, idx=None):
        if self.loss_type == 'MLE':
            return self.mle_loss(data, idx)
        elif self.loss_type == 'MSE':
            return self.mse_loss(data, idx)
        else:
            print("Please use valid loss type")
            return None

    def mle_loss(self, data, idx):
        # use indices if provided (e.g. for mini-batching)
        # otherwise, use all the data
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) == torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act = data['expert_actions'][idx]
        LL, mu, log_std = self.policy.new_dist_info(obs, act)
        # minimize negative log likelihood
        act_loss = torch.mean(mu - torch.from_numpy(act).float().to(self.device))
        return -torch.mean(LL), act_loss

    def mse_loss(self, data, idx=None):
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) is torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act_expert = data['expert_actions'][idx]
        if type(data['observations']) is not torch.Tensor:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False).to(self.device)
            act_expert = Variable(torch.from_numpy(act_expert).float().to(self.device), requires_grad=False)
        act_pi = self.policy.model(obs).squeeze()
        act_loss = torch.mean(act_pi - act_expert)
        # print("---------", act_pi.shape, act_expert.shape)
        return self.loss_criterion(act_pi, act_expert.detach()), act_loss

    def fit(self, data, suppress_fit_tqdm=False, **kwargs):
        # data is a dict
        # keys should have "observations" and "expert_actions"
        validate_keys = all([k in data.keys() for k in ["observations", "expert_actions"]])
        assert validate_keys is True
        ts = timer.time()
        num_samples = data["observations"].shape[0]

        # train loop
        global_step = 0
        for ep in config_tqdm(range(self.epochs), suppress_fit_tqdm):
            mean_loss = 0.
            inner_step = int(num_samples / self.mb_size)
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                self.optimizer.zero_grad()
                loss, act_error = self.loss(data, idx=rand_idx)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item() / inner_step
                self.logger.summary_writer.add_scalar("Training/step_loss", loss.item(), global_step=global_step)
                self.logger.summary_writer.add_scalar("Training/action_error", act_error.item(), global_step=global_step)
                global_step += 1

            self.logger.summary_writer.add_scalar("Training/mean_loss", mean_loss, global_step=ep)
            if (ep + 1) % self.eval_every == 0:
                infos = self.eval_func(self.policy)
                for k, v in infos.items():
                    self.logger.summary_writer.add_scalar(f'Eval/{k}', v, global_step=ep)

        params_after_opt = self.policy.get_param_values()
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)

    def train(self, **kwargs):
        observations = np.concatenate([path["observations"] for path in self.expert_paths])
        expert_actions = np.concatenate([path["actions"] for path in self.expert_paths])
        data = dict(observations=observations, expert_actions=expert_actions)
        self.fit(data, **kwargs)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)