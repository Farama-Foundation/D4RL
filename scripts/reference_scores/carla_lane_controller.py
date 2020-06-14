import d4rl
import gym
from d4rl.carla import data_collection_agent_lane
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='carla-lane-v0', help='Maze type. small or default')
    parser.add_argument('--num_episodes', type=int, default=100, help='Num samples to collect')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(0)
    np.random.seed(0)

    ravg = []
    for i in range(args.num_episodes):
        s = env.reset()
        controller = data_collection_agent_lane.RoamingAgent(env)
        returns = 0
        for t in range(env._max_episode_steps):
            act = controller.compute_action()

            s, rew, done, _ = env.step(act)
            returns += rew
            if done:
                break
        ravg.append(returns)
        print(i, returns, ' mean:', np.mean(ravg))
    print(args.env_name, 'returns', np.mean(ravg))


if __name__ == "__main__":
    main()

