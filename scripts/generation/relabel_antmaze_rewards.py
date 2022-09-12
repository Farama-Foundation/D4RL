import argparse
import os

import gym
import h5py
import numpy as np

from d4rl.offline_env import get_keys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="antmaze-umaze-v0", help="")
    parser.add_argument("--relabel_type", default="sparse", help="")
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    target_goal = env.target_goal
    print("Target Goal: ", target_goal)

    rdataset = h5py.File(args.filename, "r")
    fpath, ext = os.path.splitext(args.filename)
    wdataset = h5py.File(fpath + "_" + args.relabel_type + ext, "w")

    all_obs = rdataset["observations"][:]

    if args.relabel_type == "dense":
        """reward at the next state = dist(s', g)"""
        _rew = np.exp(-np.linalg.norm(all_obs[1:, :2] - target_goal, axis=1))
    elif args.relabel_type == "sparse":
        _rew = (np.linalg.norm(all_obs[1:, :2] - target_goal, axis=1) <= 0.5).astype(
            np.float32
        )
    else:
        _rew = rdataset["rewards"][:]

    # Also add terminals here
    _terminals = (np.linalg.norm(all_obs[1:, :2] - target_goal, axis=1) <= 0.5).astype(
        np.float32
    )
    _terminals = np.concatenate([_terminals, np.array([0])], 0)
    _rew = np.concatenate([_rew, np.array([0])], 0)
    print("Sum of rewards: ", _rew.sum())

    for k in get_keys(rdataset):
        print(k)
        if k == "rewards":
            wdataset.create_dataset(k, data=_rew, compression="gzip")
        elif k == "terminals":
            wdataset.create_dataset(k, data=_terminals, compression="gzip")
        else:
            wdataset.create_dataset(k, data=rdataset[k], compression="gzip")
