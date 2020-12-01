from d4rl.pointmaze import MazeEnv, maze_model
from d4rl.offline_env import get_keys
import os
import argparse
import numpy as np
import h5py


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC-BEAR')
    parser.add_argument('--maze', default='umaze', help='')
    parser.add_argument('--relabel_type', default='dense', help='')
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()


    if args.maze == 'umaze':
        maze = maze_model.U_MAZE
    elif args.maze == 'open':
        maze = maze_model.OPEN
    elif args.maze == 'medium':
        maze = maze_model.MEDIUM_MAZE
    else:
        maze = maze_model.LARGE_MAZE
    env = MazeEnv(maze, reset_target=False, reward_type='sparse')
    target_goal = env._target

    rdataset = h5py.File(args.filename, 'r')
    fpath, ext = os.path.splitext(args.filename)
    wdataset = h5py.File(fpath+'-'+args.relabel_type+ext, 'w')

    all_obs = rdataset['observations']
    if args.relabel_type == 'dense':
        _rew = np.exp(-np.linalg.norm(all_obs[:,:2] - target_goal, axis=1))
    elif args.relabel_type == 'sparse':
        _rew = (np.linalg.norm(all_obs[:,:2] - target_goal, axis=1) <= 0.5).astype(np.float32)
    else:
        _rew = rdataset['rewards'].value
    
    for k in get_keys(rdataset):
        print(k)
        if k == 'rewards':
            wdataset.create_dataset(k, data=_rew, compression='gzip')
        else:
            if k.startswith('metadata'):
                wdataset[k] = rdataset[k][()]
            else:
                wdataset.create_dataset(k, data=rdataset[k], compression='gzip')

