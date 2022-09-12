"""
Instructions:

1) Download the expert policies from https://github.com/aravindr93/hand_dapg
2) Place the policies from dapg_policies in the current directory
3) Run this script passing in the appropriate env_name
"""
import argparse
import pickle

import gym
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="", help="Environment Name")
    parser.add_argument("--num_episodes", type=int, default=100)
    args = parser.parse_args()

    policy = "./policies/" + args.env_name + ".pickle"
    pi = pickle.load(open(policy, "rb"))
    e = gym.make(args.env_name)
    e.seed(0)
    e.reset()

    ravg = []
    for n in range(args.num_episodes):
        e.reset()
        returns = 0
        for t in range(e._max_episode_steps):
            obs = e.get_obs()
            action, infos = pi.get_action(obs)
            action = pi.get_action(obs)[0]  # eval
            _, rew, done, info = e.step(action)
            returns += rew
            if done:
                break
            # e.env.mj_render() # this is much faster
            # e.render()
        ravg.append(returns)
    print(args.env_name, "returns", np.mean(ravg))


if __name__ == "__main__":
    main()
