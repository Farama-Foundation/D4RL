import brenvs
import click 
import h5py
import os
import gym
import numpy as np
import pickle
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
@click.option('--num_trajs', type=int, help='Num trajectories', default=5000)
def main(env_name, num_trajs):
    e = GymEnv(env_name)
    # render policy
    pol_playback(env_name, num_trajs)

def pol_playback(env_name, num_trajs=100):
    e = GymEnv(env_name)
    e.reset()

    obs_ = []
    act_ = []
    rew_ = []
    term_ = []
    timeout_ = []
    info_qpos_ = []
    info_qvel_ = []
    info_env_state_ = []

    ravg = []
    
    for n in range(num_trajs):
        e.reset()
        returns = 0
        for t in range(e._horizon):
            obs = e.get_obs()
            obs_.append(obs)
            info_qpos_.append(e.env.data.qpos.ravel().copy())
            info_qvel_.append(e.env.data.qvel.ravel().copy())
            info_env_state_.append(e.get_env_state())
            action = e.action_space.sample()
            act_.append(action)

            _, rew, done, info = e.step(action)
            returns += rew
            rew_.append(rew)

            if t == (e._horizon-1):
                timeout = True
                done = False
            else:
                timeout = False

            term_.append(done)
            timeout_.append(timeout)

            if done or timeout:
                e.reset()

            #e.env.mj_render() # this is much faster
            # e.render()
        ravg.append(returns)

    # write out hdf5 file
    obs_ = np.array(obs_).astype(np.float32)
    act_ = np.array(act_).astype(np.float32)
    rew_ = np.array(rew_).astype(np.float32)
    term_ = np.array(term_).astype(np.bool_)
    timeout_ = np.array(timeout_).astype(np.bool_)
    info_qpos_ = np.array(info_qpos_).astype(np.float32)
    info_qvel_ = np.array(info_qvel_).astype(np.float32)

    dataset = h5py.File('%s_random.hdf5' % env_name, 'w')

    #dataset.create_dataset('observations', obs_.shape, dtype='f4')
    dataset.create_dataset('observations', data=obs_, compression='gzip')
    dataset.create_dataset('actions', data=act_, compression='gzip')
    dataset.create_dataset('rewards', data=rew_, compression='gzip')
    dataset.create_dataset('terminals', data=term_, compression='gzip')
    dataset.create_dataset('timeouts', data=timeout_, compression='gzip')
    dataset.create_dataset('infos/qpos', data=info_qpos_, compression='gzip')
    dataset.create_dataset('infos/qvel', data=info_qvel_, compression='gzip')
    dataset.create_dataset('infos/env_state', data=np.array(info_env_state_, dtype=np.float32), compression='gzip')

if __name__ == '__main__':
    main()

