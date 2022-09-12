"""
This script runs sanity checks all datasets in a directory.

Usage:

python check_antmaze_datasets.py <dirname>
"""
import argparse
import os

import h5py
import numpy as np
import scipy as sp
import scipy.spatial


def check_identical_values(dset):
    """Check that values are not identical"""
    check_keys = ["actions", "observations", "infos/qpos", "infos/qvel"]

    for k in check_keys:
        values = dset[k][:]

        values_0 = values[0]
        values_mid = values[values.shape[0] // 2]
        values_last = values[-1]
        values = np.c_[values_0, values_mid, values_last].T
        dists = sp.spatial.distance.pdist(values)
        not_same = dists > 0
        assert np.all(not_same)


def check_num_samples(dset):
    """Check that all keys have the same # samples"""
    check_keys = [
        "actions",
        "observations",
        "rewards",
        "timeouts",
        "terminals",
        "infos/qpos",
        "infos/qvel",
    ]

    N = None
    for k in check_keys:
        values = dset[k]
        if N is None:
            N = values.shape[0]
        else:
            assert values.shape[0] == N


def check_reset_nonterminal(dataset):
    """Check if a reset occurred on a non-terminal state."""
    positions = dataset["observations"][:-1, 0:2]
    next_positions = dataset["observations"][1:, 0:2]
    diffs = np.linalg.norm(positions - next_positions, axis=1)
    terminal = ((dataset["terminals"][:] + dataset["timeouts"][:]) > 0)[:-1]

    num_resets = np.sum(diffs > 5.0)
    num_nonterminal_reset = np.sum((diffs > 5.0) * (1 - terminal))

    print("num reset:", num_resets)
    print("nonreset term:", num_nonterminal_reset)

    assert num_nonterminal_reset == 0


def print_avg_returns(dset):
    """Print returns for manual sanity checking."""
    rew = dset["rewards"][:]
    terminals = dset["terminals"][:]
    timeouts = dset["timeouts"][:]
    end_episode = (timeouts + terminals) > 0

    all_returns = []
    returns = 0
    for i in range(rew.shape[0]):
        returns += float(rew[i])
        if end_episode[i]:
            all_returns.append(returns)
            returns = 0
    print("Avg returns:", np.mean(all_returns))
    print("# timeout:", np.sum(timeouts))
    print("# terminals:", np.sum(terminals))


CHECK_FNS = [
    print_avg_returns,
    check_reset_nonterminal,
    check_identical_values,
    check_num_samples,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", type=str, help="Directory containing HDF5 datasets")
    args = parser.parse_args()
    dirname = args.dirname
    for fname in os.listdir(dirname):
        if fname.endswith(".hdf5"):
            hfile = h5py.File(os.path.join(dirname, fname))
            print("Checking:", fname)
            for check_fn in CHECK_FNS:
                try:
                    check_fn(hfile)
                except AssertionError:
                    print("Failed test:", check_fn.__name__)
                    # raise e
