import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze_bullet import bullet_maze
from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import gzip
import h5py
import argparse
import time


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, tgt, done, timeout, robot):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(False)
    data['timeouts'].append(False)
    data['infos/goal'].append(tgt)
    data['infos/goal_reached'].append(done)
    data['infos/goal_timeout'].append(timeout)
    data['infos/qpos'].append(robot.qpos.copy())
    data['infos/qvel'].append(robot.qvel.copy())

def npify(data):
    for k in data:
        if k == 'terminals' or k == 'timeouts':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    # default: p=10, d=-1
    controller = waypoint_controller.WaypointController(maze, p_gain=10.0, d_gain=-2.0)
    env = bullet_maze.Maze2DBulletEnv(maze)
    if args.render:
        env.render('human')

    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    timeout = False

    data = reset_data()
    last_position = s[0:2]
    ts = 0
    for _ in range(args.num_samples):
        position = s[0:2]
        velocity = s[2:4]

        # subtract 1.0 due to offset between tabular maze representation and bullet state
        act, done = controller.get_action(position , velocity, env._target)
        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            timeout = True
        append_data(data, s, act, env._target, done, timeout, env.robot)

        ns, _, _, _ = env.step(act)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            env.set_target()
            done = False
            ts = 0
        else:
            last_position = s[0:2]
            s = ns

        if args.render:
            env.render('human')

    
    if args.noisy:
        fname = '%s-noisy-bullet.hdf5' % args.env_name
    else:
        fname = '%s-bullet.hdf5' % args.env_name
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()
