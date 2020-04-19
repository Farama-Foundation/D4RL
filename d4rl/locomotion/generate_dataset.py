import numpy as np
import pickle
import gzip
import h5py
import argparse
from d4rl.locomotion import maze_env, ant, swimmer
from d4rl.locomotion.wrappers import NormalizedBoxEnv
from rlkit.torch.pytorch_util import set_gpu_mode
import torch
import skvideo.io
from PIL import Image
import os


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, r, tgt, done, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def load_policy(policy_file):
    data = torch.load(policy_file)
    policy = data['exploration/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    if True:
        set_gpu_mode(True)
        policy.cuda()
    return policy, env

def save_video(save_dir, file_name, frames, episode_id=0):
    filename = os.path.join(save_dir, file_name+ '_episode_{}'.format(episode_id))
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(filename, 'frame_{}.png'.format(i)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--maze', type=str, default='u-maze', help='Maze type. small or default')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    parser.add_argument('--env', type=str, default='Ant', help='Environment type')
    parser.add_argument('--policy_file', type=str, default='policy_file', help='file_name')
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--multi_start', action='store_true')
    parser.add_argument('--multigoal', action='store_true')
    args = parser.parse_args()

    if args.maze == 'u-maze':
        maze = maze_env.U_MAZE
    elif args.maze == 'big-maze':
        maze = maze_env.BIG_MAZE
    elif args.maze == 'hardest-maze':
        maze = maze_env.HARDEST_MAZE
    else:
        raise NotImplementedError
    
    if args.env == 'Ant':
        env = NormalizedBoxEnv(ant.AntMazeEnv(maze_map=maze, maze_size_scaling=4.0, non_zero_reset=args.multi_start))
    elif args.env == 'Swimmer':
        env = NormalizedBoxEnv(swimmer.SwimmerMazeEnv(mmaze_map=maze, maze_size_scaling=4.0, non_zero_reset=args.multi_start))
    
    env.set_target_goal()
    s = env.reset()
    print (s.shape)
    act = env.action_space.sample()
    done = False

    # Load the policy
    policy, train_env = load_policy(args.policy_file)

    # Define goal reaching policy fn
    def _goal_reaching_policy_fn(obs, goal):
        goal_x, goal_y = goal
        obs_new = obs[2:-2]
        goal_tuple = np.array([goal_x, goal_y])

        # normalize the norm of the relative goals to in-distribution values
        goal_tuple = goal_tuple / np.linalg.norm(goal_tuple) * 10.0

        new_obs = np.concatenate([obs_new, goal_tuple], -1)
        return policy.get_action(new_obs)[0], (goal_tuple[0] + obs[0], goal_tuple[1] + obs[1])      

    data = reset_data()

    # create waypoint generating policy integrated with high level controller
    data_collection_policy = env.create_navigation_policy(
        _goal_reaching_policy_fn,
    )

    if args.video:
        frames = []
    
    ts = 0
    num_episodes = 0
    for _ in range(args.num_samples):
        act, waypoint_goal = data_collection_policy(s)

        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.2
            act = np.clip(act, -1.0, 1.0)

        ns, r, done, info = env.step(act)
        if ts >= args.max_episode_steps:
            done = True
        
        append_data(data, s[:-2], act, r, env.target_goal, done, env.physics.data)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1

        if done:
            done = False
            ts = 0
            s = env.reset()
            env.set_target_goal()
            if args.video:
                frames = np.array(frames)
                save_video('./videos/', args.env + '_navigation', frames, num_episodes)
            
            num_episodes += 1
            frames = []
        else:
            s = ns

        if args.video:
            curr_frame = env.physics.render(width=500, height=500, depth=False)
            frames.append(curr_frame)
    
    if args.noisy:
        fname = args.env + '_maze_%s_noisy_multistart_%s_multigoal_%s.hdf5' % (args.maze, str(args.multi_start), str(args.multigoal))
    else:
        fname = args.env + 'maze_%s_multistart_%s_multigoal_%s.hdf5' % (args.maze, str(args.multi_start), str(args.multigoal))
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

if __name__ == '__main__':
    main()
