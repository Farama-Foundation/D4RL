import argparse

import gym
import numpy as np

from d4rl.pointmaze import waypoint_controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="maze2d-umaze-v0",
        help="Maze type. small or default",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100, help="Num samples to collect"
    )
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(0)
    np.random.seed(0)
    controller = waypoint_controller.WaypointController(env.str_maze_spec)

    ravg = []
    for _ in range(args.num_episodes):
        s = env.reset()
        returns = 0
        for t in range(env._max_episode_steps):
            position = s[0:2]
            velocity = s[2:4]
            act, done = controller.get_action(position, velocity, env.get_target())
            s, rew, _, _ = env.step(act)
            returns += rew
        ravg.append(returns)
    print(args.env_name, "returns", np.mean(ravg))


if __name__ == "__main__":
    main()
