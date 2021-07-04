import argparse
import re

import h5py
import torch
import gym
import d4rl
import numpy as np

from rlkit.torch import pytorch_util as ptu

itr_re = re.compile(r'itr_(?P<itr>[0-9]+).pkl')

def load(pklfile):
    params = torch.load(pklfile)
    return params['trainer/policy']

def get_pkl_itr(pklfile):
    match = itr_re.search(pklfile)
    if match:
        return match.group('itr')
    raise ValueError(pklfile+" has no iteration number.")

def get_policy_wts(params):
    out_dict = {
        'fc0/weight': params.fcs[0].weight.data.numpy(),
        'fc0/bias': params.fcs[0].bias.data.numpy(),
        'fc1/weight': params.fcs[1].weight.data.numpy(),
        'fc1/bias': params.fcs[1].bias.data.numpy(),
        'last_fc/weight': params.last_fc.weight.data.numpy(),
        'last_fc/bias': params.last_fc.bias.data.numpy(),
        'last_fc_log_std/weight': params.last_fc_log_std.weight.data.numpy(),
        'last_fc_log_std/bias': params.last_fc_log_std.bias.data.numpy(),
    }
    return out_dict

def get_reset_data():
    data = dict(
        observations = [],
        next_observations = [],
        actions = [],
        rewards = [],
        terminals = [],
        timeouts = [],
        logprobs = [],
        qpos = [],
        qvel = []
    )
    return data

def rollout(policy, env_name, max_path, num_data, random=False):
    env = gym.make(env_name)

    data = get_reset_data()
    traj_data = get_reset_data()

    _returns = 0
    t = 0 
    done = False
    s = env.reset()
    while len(data['rewards']) < num_data:


        if random:
            a = env.action_space.sample()
            logprob = np.log(1.0 / np.prod(env.action_space.high - env.action_space.low))
        else:
            torch_s = ptu.from_numpy(np.expand_dims(s, axis=0))
            distr = policy.forward(torch_s)
            a = distr.sample()
            logprob = distr.log_prob(a)
            a = ptu.get_numpy(a).squeeze()

        #mujoco only
        qpos, qvel = env.sim.data.qpos.ravel().copy(), env.sim.data.qvel.ravel().copy()

        try:
            ns, rew, done, infos = env.step(a)
        except:
            print('lost connection')
            env.close()
            env = gym.make(env_name)
            s = env.reset()
            traj_data = get_reset_data()
            t = 0
            _returns = 0
            continue

        _returns += rew

        t += 1
        timeout = False
        terminal = False
        if t == max_path:
            timeout = True
        elif done:
            terminal = True


        traj_data['observations'].append(s)
        traj_data['actions'].append(a)
        traj_data['next_observations'].append(ns)
        traj_data['rewards'].append(rew)
        traj_data['terminals'].append(terminal)
        traj_data['timeouts'].append(timeout)
        traj_data['logprobs'].append(logprob)
        traj_data['qpos'].append(qpos)
        traj_data['qvel'].append(qvel)

        s = ns
        if terminal or timeout:
            print('Finished trajectory. Len=%d, Returns=%f. Progress:%d/%d' % (t, _returns, len(data['rewards']), num_data))
            s = env.reset()
            t = 0
            _returns = 0
            for k in data:
                data[k].extend(traj_data[k])
            traj_data = get_reset_data()
    
    new_data = dict(
        observations=np.array(data['observations']).astype(np.float32),
        actions=np.array(data['actions']).astype(np.float32),
        next_observations=np.array(data['next_observations']).astype(np.float32),
        rewards=np.array(data['rewards']).astype(np.float32),
        terminals=np.array(data['terminals']).astype(np.bool),
        timeouts=np.array(data['timeouts']).astype(np.bool)
    )
    new_data['infos/action_log_probs'] = np.array(data['logprobs']).astype(np.float32)
    new_data['infos/qpos'] = np.array(data['qpos']).astype(np.float32)
    new_data['infos/qvel'] = np.array(data['qvel']).astype(np.float32)

    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--pklfile', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='output.hdf5')
    parser.add_argument('--max_path', type=int, default=1000)
    parser.add_argument('--num_data', type=int, default=10000)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = None
    if not args.random:
        policy = load(args.pklfile)
    data = rollout(policy, args.env, max_path=args.max_path, num_data=args.num_data, random=args.random)

    hfile = h5py.File(args.output_file, 'w')
    for k in data:
        hfile.create_dataset(k, data=data[k], compression='gzip')

    if args.random:
        pass
    else:
        hfile['metadata/algorithm'] = np.string_('SAC')
        hfile['metadata/iteration'] = np.array([get_pkl_itr(args.pklfile)], dtype=np.int32)[0]
        hfile['metadata/policy/nonlinearity'] = np.string_('relu')
        hfile['metadata/policy/output_distribution'] = np.string_('tanh_gaussian')
        for k, v in get_policy_wts(policy).items():
            hfile['metadata/policy/'+k] = v
    hfile.close()
