import torch
import numpy as np

from torch import nn

from torch.distributions import Normal

class BC(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, n_emb: int = 256, device="cpu") -> None:
        super().__init__()
        self.is_discrete = False
        self.action_dim = action_dim
        self.observation_dim = observation_dim

        self.emb = nn.Sequential(
            nn.LayerNorm(observation_dim),
            nn.Linear(observation_dim, n_emb),
            nn.GELU(),
            nn.Linear(n_emb, n_emb),
            nn.GELU(),
            nn.Linear(n_emb, action_dim * 2),
        )
        self.device = device

    def get_action(self, obs, action_mask):
        with torch.no_grad():
            obs = torch.from_numpy(obs.reshape(1, -1)).float().to(self.device)
            logits = self.emb(obs)
            mu = logits[..., 0]
            scale = logits[..., 1]
            dist = Normal(mu, scale.exp())
            ava_actions = np.asarray([idx for idx, v in enumerate(action_mask) if v > 0])
            ava_actions = torch.from_numpy(ava_actions).float().to(self.device)
            log_probs = dist.log_prob(ava_actions)
            ava_idx = log_probs.argmax()
            action = ava_actions[ava_idx].long().item()
        return action, log_probs[ava_idx].item()

    def forward(self, observation: torch.Tensor):
        batch_size = observation.shape[0]
        x = self.emb(observation).view(batch_size, self.action_dim, 2)
        return x
