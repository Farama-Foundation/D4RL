import argparse
import h5py
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file1', type=str, default=None)
    parser.add_argument('file2', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='output.hdf5')
    parser.add_argument('--maxlen', type=int, default=2000000)
    args = parser.parse_args()

    hfile1 = h5py.File(args.file1, 'r')
    hfile2 = h5py.File(args.file2, 'r')
    outf = h5py.File(args.output_file, 'w')

    keys = ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'timeouts', 'infos/action_log_probs', 'infos/qpos', 'infos/qvel']
    # be careful with trajectories not ending at the end of a file!
    
    # find end of last traj
    terms = hfile1['terminals'][:]
    tos = hfile1['timeouts'][:]
    last_term = 0
    for i in range(terms.shape[0]-1, -1, -1):
        if terms[i] or tos[i]:
            last_term = i
            break
    N = last_term + 1

    for k in keys:
        d1 = hfile1[k][:N]
        d2 = hfile2[k][:]
        combined = np.concatenate([d1,d2],axis=0)[:args.maxlen]
        print(k, combined.shape)
        outf.create_dataset(k, data=combined, compression='gzip')

    outf.close()
