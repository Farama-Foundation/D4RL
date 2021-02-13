import logging
from offline_rl.gym_minigrid import fourroom_controller
from offline_rl.gym_minigrid.envs import fourrooms
import numpy as np
import pickle
import gzip
import h5py
import argparse


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/pos': [],
            'infos/orientation': [],
            }

def append_data(data, s, a, tgt, done, pos, ori):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/pos'].append(pos)
    data['infos/orientation'].append(ori)

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--random', action='store_true', help='Noisy actions')
    parser.add_argument('--num_samples', type=int, default=int(1e5), help='Num samples to collect')
    args = parser.parse_args()

    controller = fourroom_controller.FourRoomController()
    env = fourrooms.FourRoomsEnv()

    controller.set_target(controller.sample_target())
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts = 0
    for _ in range(args.num_samples):
        if args.render:
            env.render()

        if args.random:
            act = env.action_space.sample()
        else:
            act, done = controller.get_action(env.agent_pos, env.agent_dir) 

        if ts >= 50:
            done = True
        append_data(data, s['image'], act, controller.target, done, env.agent_pos, env.agent_dir)

        ns, _, _, _ = env.step(act)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            controller.set_target(controller.sample_target())
            done = False
            ts = 0
        else:
            s = ns
    
    if args.random:
        fname = 'minigrid4rooms_random.hdf5'
    else:
        fname = 'minigrid4rooms.hdf5' 
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()
