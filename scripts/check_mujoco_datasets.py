"""
This script runs sanity checks all datasets in a directory.
Assumes all datasets in the directory are generated via mujoco and contain
the qpos/qvel keys.

Usage:

python check_mujoco_datasets.py <dirname>
"""
import numpy as np
import scipy as sp
import scipy.spatial
import h5py
import os
import argparse
import tqdm


def check_identical_values(dset):
    """ Check that values are not identical """
    check_keys = ['actions', 'observations', 'infos/qpos', 'infos/qvel']

    for k in check_keys:
        values = dset[k][:]

        values_0 = values[0]
        values_mid = values[values.shape[0]//2]
        values_last = values[-1]
        values = np.c_[values_0, values_mid, values_last].T
        dists = sp.spatial.distance.pdist(values)
        not_same = dists > 0
        assert np.all(not_same)


def check_qpos_qvel(dset):
    """ Check that qpos/qvel produces correct state"""
    import gym
    import d4rl

    N = dset['rewards'].shape[0]
    qpos = dset['infos/qpos']
    qvel = dset['infos/qvel']
    obs = dset['observations']

    reverse_env_map = {v.split('/')[-1]: k for (k, v) in d4rl.infos.DATASET_URLS.items()}
    env_name = reverse_env_map[dset.filename.split('/')[-1]]
    env = gym.make(env_name)
    env.reset()
    print('checking qpos/qvel')
    for t in tqdm.tqdm(range(N)):
        env.set_state(qpos[t], qvel[t])
        env_obs = env.env.wrapped_env._get_obs()
        error = ((obs[t] - env_obs)**2).sum()
        assert error < 1e-8

def check_num_samples(dset):
    """ Check that all keys have the same # samples """
    check_keys = ['actions', 'observations', 'rewards', 'timeouts', 'terminals', 'infos/qpos', 'infos/qvel']

    N = None
    for k in check_keys:
        values = dset[k]
        if N is None:
            N = values.shape[0]
        else:
            assert values.shape[0] == N


def check_reset_state(dset):
    """ Check that resets correspond approximately to the initial state """
    obs = dset['observations'][:]
    N = obs.shape[0]
    terminals = dset['terminals'][:]
    timeouts = dset['timeouts'][:]
    end_episode = (timeouts + terminals) > 0

    # Use the first observation as a reference initial state
    reset_state = obs[0]

    # Make sure all reset observations are close to the reference initial state

    # Take up to [:-1] in case last entry in dataset is terminal
    end_idxs = np.where(end_episode)[0][:-1]

    diffs = obs[1:] - reset_state
    dists = np.linalg.norm(diffs, axis=1)

    min_dist = np.min(dists)
    reset_dists = dists[end_idxs]  #don't add idx +1 because we took the obs[:1] slice
    print('max reset:', np.max(reset_dists))
    print('min reset:', np.min(reset_dists))

    assert np.all(reset_dists < (min_dist + 1e-2) * 5)


def print_avg_returns(dset):
    """ Print returns for manual sanity checking. """
    rew = dset['rewards'][:]
    terminals = dset['terminals'][:]
    timeouts = dset['timeouts'][:]
    end_episode = (timeouts + terminals) > 0

    all_returns = []
    returns = 0
    for i in range(rew.shape[0]):
        returns += float(rew[i])
        if end_episode[i]:
            all_returns.append(returns)
            returns = 0
    print('Avg returns:', np.mean(all_returns))
    print('# timeout:', np.sum(timeouts))
    print('# terminals:', np.sum(terminals))


CHECK_FNS = [print_avg_returns, check_qpos_qvel, check_reset_state, check_identical_values, check_num_samples]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', type=str, help='Directory containing HDF5 datasets')
    args = parser.parse_args()
    dirname = args.dirname
    for fname in os.listdir(dirname):
        if fname.endswith('.hdf5'):
            hfile = h5py.File(os.path.join(dirname, fname))
            print('Checking:', fname)
            for check_fn in CHECK_FNS:
                try:
                    check_fn(hfile)
                except AssertionError as e:
                    print('Failed test:', check_fn.__name__)
                    raise e

