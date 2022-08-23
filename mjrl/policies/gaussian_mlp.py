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

    def get_action(self, obs, action_mask=None):
        with torch.no_grad():
            obs = torch.from_numpy(obs.reshape(1, -1)).float().to(self.device)
            logits = self.forward(obs)
            mu = logits[..., 0]
            scale = logits[..., 1]
            dist = Normal(mu, scale.exp())
            action = dist.sample().cpu().numpy()[0]

        return action, None

    def forward(self, observation: torch.Tensor):
        batch_size = observation.shape[0]
        x = self.emb(observation).view(batch_size, self.action_dim, 2)
        return x
