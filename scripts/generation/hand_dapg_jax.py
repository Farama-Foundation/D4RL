import d4rl
import click 
import h5py
import os
import gym
import numpy as np
import pickle
import gzip
import collections
from mjrl.utils.gym_env import GymEnv

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name relocate-v0 --policy policies/relocate-v0.pickle --mode evaluation\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--snapshot_file', type=str, help='absolute path of the policy file', required=True)
@click.option('--num_trajs', type=int, help='Num trajectories', default=5000)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
def main(env_name, snapshot_file, mode, num_trajs, clip=True):
    e = GymEnv(env_name)
    pi = pickle.load(gzip.open(snapshot_file, 'rb'))
    import pdb; pdb.set_trace()
    pass
    # render policy
    #pol_playback(env_name, pi, num_trajs, clip=clip)


def extract_params(policy):

    out_dict = {
        'fc0/weight': _fc0w,
        'fc0/bias': _fc0b,
        'fc1/weight': params[2].data.numpy(),
        'fc1/bias': params[3].data.numpy(),
        'last_fc/weight': _fclw,
        'last_fc/bias': _fclb,
        'last_fc_log_std/weight': _fclw,
        'last_fc_log_std/bias': _fclb,
    }
    return out_dict


def pol_playback(env_name, pi, num_trajs=100, clip=True):
    e = gym.make(env_name)
    e.reset()

    obs_ = []
    act_ = []
    rew_ = []
    term_ = []
    timeout_ = []
    info_qpos_ = []
    info_qvel_ = []
    info_mean_ = []
    info_logstd_ = []
    info_env_state_ = collections.defaultdict(list)

    ravg = []
    
    for n in range(num_trajs):
        e.reset()
        returns = 0
        for t in range(e._max_episode_steps):
            obs = e.get_obs()
            obs_.append(obs)
            info_qpos_.append(e.env.data.qpos.ravel().copy())
            info_qvel_.append(e.env.data.qvel.ravel().copy())
            [info_env_state_[k].append(v) for k,v in e.get_env_state().items()]
            action, infos = pi.get_action(obs)
            action = pi.get_action(obs)[0] # eval
            
            if clip:
                action = np.clip(action, -1, 1)

            act_.append(action)
            info_mean_.append(infos['mean'])
            info_logstd_.append(infos['log_std'])

            _, rew, done, info = e.step(action)
            returns += rew
            rew_.append(rew)

            if t == (e._max_episode_steps-1):
                timeout = True
                done = False
            else:
                timeout = False
            term_.append(done)
            timeout_.append(timeout)

            if done or timeout:
                e.reset()
                break

            #e.env.mj_render() # this is much faster
            # e.render()
        ravg.append(returns)
        print(n, returns, t)

    # write out hdf5 file
    obs_ = np.array(obs_).astype(np.float32)
    act_ = np.array(act_).astype(np.float32)
    rew_ = np.array(rew_).astype(np.float32)
    term_ = np.array(term_).astype(np.bool_)
    timeout_ = np.array(timeout_).astype(np.bool_)
    info_qpos_ = np.array(info_qpos_).astype(np.float32)
    info_qvel_ = np.array(info_qvel_).astype(np.float32)
    info_mean_ = np.array(info_mean_).astype(np.float32)
    info_logstd_ = np.array(info_logstd_).astype(np.float32)

    if clip:
        dataset = h5py.File('%s_expert_clip.hdf5' % env_name, 'w')
    else:
        dataset = h5py.File('%s_expert.hdf5' % env_name, 'w')

    #dataset.create_dataset('observations', obs_.shape, dtype='f4')
    dataset.create_dataset('observations', data=obs_, compression='gzip')
    dataset.create_dataset('actions', data=act_, compression='gzip')
    dataset.create_dataset('rewards', data=rew_, compression='gzip')
    dataset.create_dataset('terminals', data=term_, compression='gzip')
    dataset.create_dataset('timeouts', data=timeout_, compression='gzip')
    #dataset.create_dataset('infos/qpos', data=info_qpos_, compression='gzip')
    #dataset.create_dataset('infos/qvel', data=info_qvel_, compression='gzip')
    dataset.create_dataset('infos/action_mean', data=info_mean_, compression='gzip')
    dataset.create_dataset('infos/action_log_std', data=info_logstd_, compression='gzip')
    for k in info_env_state_:
        dataset.create_dataset('infos/%s' % k, data=np.array(info_env_state_[k], dtype=np.float32), compression='gzip')

    # write metadata
    policy_params = extract_params(pi)
    dataset['metadata/algorithm'] = np.string_('DAPG')
    dataset['metadata/policy/nonlinearity'] = np.string_('tanh')
    dataset['metadata/policy/output_distribution'] = np.string_('gaussian')
    for k, v in policy_params.items():
        dataset['metadata/policy/'+k] = v

if __name__ == '__main__':
    main()

