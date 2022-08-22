import numpy as np
import torch

from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from mjrl.utils.fc_network import FCNetwork
from mjrl.utils.gym_env import EnvSpec


class MLP:
    def __init__(self, env_spec: EnvSpec, hidden_sizes=(64,64), device="cpu") -> None:
        self.input_dim = env_spec.observation_dim
        self.output_dim = env_spec.action_dim
        self.device = device
        self.is_discrete = True

        self.model = FCNetwork(self.input_dim, self.output_dim, hidden_sizes, device=device).to(self.device)

    def parameters(self):
        return self.model.parameters()

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    def get_action(self, observation, action_mask):
        with torch.no_grad():
            action_logits = self.model(torch.from_numpy(observation).float().to(self.device)) # / 0.5
            action_mask = np.abs(action_mask - 1)
            action_logits -= torch.from_numpy(action_mask).float().to(self.device) * 1e6
            # use sample?
            # dist = Categorical(logits=action_logits)
            # action = dist.sample().cpu().numpy()
            action = action_logits.argmax(-1).cpu().numpy()
        return [action, {}]

    def old_dist_info(self, observations, actions):
        LL = self.log_likelihood(observations, actions)
        return [LL, None, None]

    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.cpu().numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float().to(self.device)
                current_idx += self.param_sizes[idx]
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float().to(self.device)
                current_idx += self.param_sizes[idx]
