"""
Generate "minimum" reference scores by averaging the score for a random
policy over 100 episodes.
"""
import argparse

import gym
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="", help="Environment Name")
    parser.add_argument("--num_episodes", type=int, default=100)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(0)
    env.action_space.seed(0)

    ravg = []
    for n in range(args.num_episodes):
        env.reset()
        returns = 0
        for t in range(env._max_episode_steps):
            action = env.action_space.sample()
            _, rew, done, info = env.step(action)
            returns += rew
            if done:
                break
        ravg.append(returns)
    print(
        "%s Average returns (%d ep): %f"
        % (args.env_name, args.num_episodes, np.mean(ravg))
    )


if __name__ == "__main__":
    main()
