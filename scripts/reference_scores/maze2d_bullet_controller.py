import d4rl
import gym
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v0', help='Maze type. small or default')
    parser.add_argument('--num_episodes', type=int, default=100, help='Num samples to collect')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    if args.render:
        env.render('human')
    env.seed(0)
    np.random.seed(0)
    d_gain = -2.0
    p_gain = 10.0
    controller = waypoint_controller.WaypointController(env.env.str_maze_spec, p_gain=p_gain, d_gain=d_gain)
    print('max steps:', env._max_episode_steps)

    ravg = []
    for _ in range(args.num_episodes):
        controller = waypoint_controller.WaypointController(env.env.str_maze_spec, p_gain=p_gain, d_gain=d_gain)
        s = env.reset()
        returns = 0
        for t in range(env._max_episode_steps):
            position = s[0:2] 
            velocity = s[2:4]
            act, done = controller.get_action(position, velocity, np.array(env.env.get_target()))
            #print(position-1, controller.current_waypoint(), np.array(env.env.get_target()) - 1)
            #print('\t', act)
            s, rew, _, _ = env.step(act)
            if args.render:
                time.sleep(0.01)
                env.render('human')
            returns += rew
        print(returns)
        ravg.append(returns)
    print(args.env_name, 'returns', np.mean(ravg))


if __name__ == "__main__":
    main()
