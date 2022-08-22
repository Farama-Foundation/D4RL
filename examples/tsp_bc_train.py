import gym
import d4rl
import tqdm
import numpy as np
import torch
import os
import time
from torch.nn import functional as F

from mjrl.algos.behavior_cloning_2 import BC
from mjrl.utils.gym_env import EnvSpec
from mjrl.policies.discrete_mlp import MLP
# from mjrl.policies.gaussian_mlp import BC as MLP


device = "cuda:1"

env = gym.make("tsp200-1-expert-v1")
dataset = env.get_dataset()
obs_dim = dataset["observations"].shape[-1]
env_spec = EnvSpec(obs_dim=obs_dim, act_dim=env.action_dim, horizon=100)

print("env obs dim and action dim:", env.state_dim, env.action_dim)

policy = MLP(env_spec, device=device, hidden_sizes=(256, 256))


def test_target():
    obs = env.reset()
    done = False
    actions = env.dataset.soln[env.sequence_idx][1:]
    step = 0
    total_rew = 0
    while not done:
        node_idx = env.get_cur_ava_nodes().tolist().index(actions[step])
        assert node_idx < env.action_dim
        obs, rew, done, info = env.step(node_idx)
        step += 1
        total_rew += rew
    print("total reward:", total_rew)



def eval_func(policy, print_traj=False):
    n_episodes = 5
    score = 0
    mean_return = 0
    for episode in tqdm.tqdm(range(n_episodes), desc="Evaluation", leave=False):
        done = False
        traj_idx = np.random.choice(10000)
        obs = env.reset(traj_idx=traj_idx)
        total_rew = 0
        step = 0
        while not done:
            action_mask = env.get_cur_action_mask()
            action = policy.get_action(obs, action_mask)[0]
            obs, rew, done, info = env.step(action)
            total_rew += rew
            step += 1
        score += env.get_normalized_score(total_rew) / n_episodes
        mean_return += total_rew / n_episodes
    # if print_traj:
    #     real_traj  env.dataset.soln[env.sequence_idx][1:].astype(np.int32).tolist()
    #     print(f"* action_hist: {action_hist}\n* real_hist: {real_traj}\n* len vs: {len(action_hist)} / {len(real_traj)}")

    return {"score": score, "episode_reward": mean_return}


# test eval
test_target()
print(eval_func(policy))

log_dir = f"./logs/{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)


bc = BC(expert_paths=[dataset], policy=policy, loss_type='MLE', set_transforms=False, eval_every=1, epochs=10000, eval_func=eval_func, device=device, lr=1e-4, log_dir=log_dir)
bc.train()