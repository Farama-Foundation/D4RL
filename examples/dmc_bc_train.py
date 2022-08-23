import argparse
import time
import os

import d4rl
import gym
import tqdm
import numpy as np

from d4rl.dmc.dataset_info import DATASET_URLS
from mjrl.algos.behavior_cloning_2 import BC
from mjrl.utils.gym_env import EnvSpec

from mjrl.policies.discrete_mlp import MLP as DiscreteMLP
from mjrl.policies.gaussian_mlp import BC as GaussianMLP


def eval_func(policy, empty=None):
    n_episodes = 5
    score = 0
    mean_return = 0

    for _ in tqdm.tqdm(range(n_episodes), desc="Evaluation", leave=False):
        done = False
        total_rew = 0
        step = 0
        env.seed(np.random.choice(100))
        obs = env.reset()
        while not done:
            action = policy.get_action(obs)[0]
            obs, rewa, done, info = env.step(action)
            total_rew += rewa
            step += 1
        score += env.get_normalized_score(total_rew) / n_episodes
        mean_return += total_rew / n_episodes
    return {"score": score, "episode_reward": mean_return}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--scenario", type=str, required=True)

    args = parser.parse_args()
    scenarios = list(DATASET_URLS)

    for scenario in scenarios:
        env = gym.make(f"dmc-{scenario}-expert-v1")
        dataset = env.get_dataset()
        obs_dim = dataset["observations"].shape[-1]
        env_spec = EnvSpec(obs_dim, act_dim=env.action_space.shape[0], horizon=1000)

        if env.is_discrete:
            policy = DiscreteMLP(env_spec, device=args.device, hidden_sizes=(256, 256))
        else:
            policy = GaussianMLP(obs_dim, env.action_space.shape[0], device=args.device,).to(args.device)

        print(eval_func(policy))

        log_dir = f"./logs/{scenario}/{int(time.time())}"
        os.makedirs(log_dir, exist_ok=True)

        bc = BC(expert_paths=[dataset], policy=policy, loss_type='MSE', set_transforms=False, eval_every=5000, epochs=20, eval_func=eval_func, device=args.device, lr=1e-3, log_dir=log_dir)
        bc.train(max_steps=100000)

