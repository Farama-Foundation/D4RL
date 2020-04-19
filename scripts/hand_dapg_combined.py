import gym
import d4rl
import argparse
import os
import numpy as np
import h5py

def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--env_name', type=str, default='pen', help='Env name')
    args = parser.parse_args()

    env = gym.make('%s-v0' % args.env_name)
    human_dataset = gym.make('%s-human-v0' % args.env_name).get_dataset()
    bc_dataset = gym.make('%s-demos-v0' % args.env_name).get_dataset()
    N = env._max_episode_steps * 5000
    halfN = N // 2

    aug_dataset = h5py.File('%s-demos-v0-bc-combined.hdf5' % args.env_name, 'w')
    for k in human_dataset:
        human_data = human_dataset[k]
        bc_data = bc_dataset[k][:halfN]
        print(k, human_data.shape, bc_data.shape)
        N_tile = int(halfN / human_data.shape[0]) + 1
        if len(human_data.shape) == 1:
            human_data = np.tile(human_data, [N_tile])[:halfN]
        elif len(human_data.shape) == 2:
            human_data = np.tile(human_data, [N_tile, 1])[:halfN]
        else:
            raise NotImplementedError()

        # clone demo_data
        aug_data = np.concatenate([human_data, bc_data], axis=0)
        assert aug_data.shape[1:] == bc_data.shape[1:]
        assert aug_data.shape[1:] == human_data.shape[1:]

        print('\t',human_data.shape, bc_data.shape, '->',aug_data.shape)
        aug_dataset.create_dataset(k, data=aug_data, compression='gzip')

