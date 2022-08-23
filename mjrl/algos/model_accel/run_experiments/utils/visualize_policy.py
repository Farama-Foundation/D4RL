import gym
import mjrl.envs
import trajopt.envs
import mj_envs
import click
import os
import gym
import numpy as np
import pickle
import torch
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
import trajopt.envs

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name mjrl_swimmer-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=10)
@click.option('--log_std', type=float, default=-0.5)
@click.option('--terminate', type=bool, default=True)
@click.option('--device_path', type=str, default=None)
def main(env_name, policy, mode, seed, episodes, log_std, terminate, device_path):
    render = True

    # TODO(Aravind): Map to hardware if device_path is specified

    e = GymEnv(env_name)
    e.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if policy is not None:
        policy = pickle.load(open(policy, 'rb'))
    else:
        policy = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=log_std)

    for ep in range(episodes):
        o = e.reset()
        rew = 0.0
        t = 0
        done = False
        while t < e.horizon and done is False:
            o = e.get_obs()
            a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
            next_o, r, done, ifo = e.step(a)
            if terminate is False:
                done = False
            rew = rew + r
            t = t + 1
            if render:
                e.render()
            if done and t < e.horizon - 1:
                print("Episode terminated early")
        print("episode score = %f " % rew)

    e.reset()


if __name__ == '__main__':
    main()
