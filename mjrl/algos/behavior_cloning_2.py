"""
Minimize bc loss (MLE, MSE, RWR etc.) with pytorch optimizers
"""

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch

from torch.distributions import Normal

from torch.nn import functional as F
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
from tqdm import tqdm


class BC:
    def __init__(self, expert_paths,
                 policy,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 loss_type = 'MSE',  # can be 'MLE' or 'MSE'
                 log_dir="./logs",
                 **kwargs,
                 ):

        self.policy = policy
        self.expert_paths = expert_paths
        self.epochs = epochs
        self.mb_size = batch_size
        self.logger = DataLog(log_dir)
        self.loss_type = loss_type
        self.eval_every = kwargs["eval_every"]
        self.eval_func = kwargs["eval_func"]
        self.device = kwargs["device"]

        # construct optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Loss criterion if required
        if loss_type == 'MSE':
            self.loss_criterion = torch.nn.MSELoss()

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
        length = len(data["observations"])
        idx = idx if idx is not None else np.random.choice(length, self.mb_size)
        obs = data['observations'][idx]
        act = data['expert_actions'][idx]

        batch_size = obs.shape[0]

        obs = torch.from_numpy(obs).float().to(self.device)
        action_logits = self.policy(obs)

        if not self.policy.is_discrete:
            act = torch.from_numpy(act).float().to(self.device).reshape(batch_size, -1)
            mu = action_logits[..., 0]
            scale = action_logits[..., 1]
            zs = (act - mu) / torch.exp(scale)
            LL = torch.mean(-0.5 * torch.sum(zs ** 2, dim=1) - torch.sum(scale, dim=1) - 0.5 * mu * np.log(2 * np.pi))
            action_error = torch.mean(torch.abs(act - mu))
        else:
            # add mask here
            # mask = torch.abs(torch.from_numpy(data['ava_actions'][idx]).float().to(self.device) - 1.) * 1e6
            # action_logits += mask

            policy = F.softmax(action_logits, dim=-1)
            actions = torch.from_numpy(act).long().to(action_logits.device)
            # print("max actions", self.policy.output_dim, actions.max(), actions.min())
            action_error = torch.mean(torch.abs(policy.argmax(-1) - actions.float())).detach()
            LL = F.cross_entropy(policy, actions)
        return LL, action_error

    def mse_loss(self, data, idx=None):
        length = len(data["observations"])
        idx = idx if idx is not None else np.random.choice(length, self.mb_size)
        obs = data['observations'][idx]
        act_expert = data['expert_actions'][idx]
        obs = torch.from_numpy(obs).float().to(self.device)
        act_expert = torch.from_numpy(act_expert).to(self.device)
        
        if not self.policy.is_discrete:
            act_logits = self.policy(obs)
            mu = act_logits[..., 0]
            scale = act_logits[..., 1]
            dist = Normal(mu, scale.exp())
            actions = dist.rsample()
            act_expert = act_expert.float().reshape(self.mb_size, -1)
            assert act_expert.shape == actions.shape
            action_error = torch.mean(torch.abs(actions - act_expert)).detach()
        else:
            act_pi = self.policy.model(obs)
            actions = F.softmax(act_pi, dim=-1)
            action_error = torch.mean(torch.abs(act_pi.argmax(-1) - act_expert.float())).detach()
            act_expert = F.one_hot(act_expert.long(), 100).float()

        return F.mse_loss(actions, act_expert), action_error

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
            for mb in tqdm(range(inner_step), desc="Epoch", leave=False):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                self.optimizer.zero_grad()
                loss, act_error = self.loss(data, idx=rand_idx)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item() / inner_step
                self.logger.summary_writer.add_scalar("Training/step_loss", loss.item(), global_step=global_step)
                self.logger.summary_writer.add_scalar("Training/action_error", act_error.item(), global_step=global_step)
                if (global_step + 1) % self.eval_every == 0:
                    infos = self.eval_func(self.policy, (global_step + 1) % 100 == 0)
                    for k, v in infos.items():
                        self.logger.summary_writer.add_scalar(f'Eval/{k}', v, global_step=global_step)
                global_step += 1
            self.logger.summary_writer.add_scalar("Training/mean_loss", mean_loss, global_step=ep)

    def train(self, **kwargs):
        observations = np.concatenate([path["observations"] for path in self.expert_paths])
        expert_actions = np.concatenate([path["actions"] for path in self.expert_paths])
        ava_actions = np.concatenate([path['ava_actions'] for path in self.expert_paths])
        data = dict(observations=observations, expert_actions=expert_actions, ava_actions=ava_actions)
        self.fit(data, **kwargs)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)