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
    parser.add_argument('--bc', type=str, help='BC hdf5 dataset')
    parser.add_argument('--human', type=str, help='Human demos hdf5 dataset')
    args = parser.parse_args()

    env = gym.make('%s-v0' % args.env_name)
    human_dataset = h5py.File(args.human, 'r')
    bc_dataset = h5py.File(args.bc, 'r')
    N = env._max_episode_steps * 5000

    # search for nearest terminal after the halfway mark
    halfN = N // 2
    terms = bc_dataset['terminals'][:]
    tos = bc_dataset['timeouts'][:]
    last_term = 0
    for i in range(halfN, N):
        if terms[i] or tos[i]:
            last_term = i
            break
    halfN = last_term + 1

    remaining_N = N - halfN

    aug_dataset = h5py.File('%s-cloned-v1.hdf5' % args.env_name, 'w')
    for k in get_keys(bc_dataset):
        if 'metadata' not in k:
            human_data = human_dataset[k][:]
            bc_data = bc_dataset[k][:halfN]
            print(k, human_data.shape, bc_data.shape)
            N_tile = int(halfN / human_data.shape[0]) + 1
            if len(human_data.shape) == 1:
                human_data = np.tile(human_data, [N_tile])[:remaining_N]
            elif len(human_data.shape) == 2:
                human_data = np.tile(human_data, [N_tile, 1])[:remaining_N]
            else:
                raise NotImplementedError()

            # clone demo_data
            aug_data = np.concatenate([bc_data, human_data], axis=0)
            assert aug_data.shape[1:] == bc_data.shape[1:]
            assert aug_data.shape[1:] == human_data.shape[1:]

            print('\t',human_data.shape, bc_data.shape, '->',aug_data.shape)
            aug_dataset.create_dataset(k, data=aug_data, compression='gzip')
        else:
            shape = bc_dataset[k].shape
            print('metadata:', k, shape)
            if len(shape) == 0:
                aug_dataset[k] = bc_dataset[k][()]
            else:
                aug_dataset[k] = bc_dataset[k][:]

