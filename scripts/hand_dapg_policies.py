import d4rl
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
#@click.option('--policy', type=str, help='absolute path of the policy file', required=True)
@click.option('--num_trajs', type=int, help='Num trajectories', default=5000)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
def main(env_name, mode, num_trajs, clip=True):
    e = GymEnv(env_name)
    policy = './policies/'+env_name+'.pickle'
    pi = pickle.load(open(policy, 'rb'))
    # render policy
    pol_playback(env_name, pi, num_trajs, clip=clip)

def pol_playback(env_name, pi, num_trajs=100, clip=True):
    e = gym.make(env_name)
    e.reset()

    obs_ = []
    act_ = []
    rew_ = []
    term_ = []
    info_qpos_ = []
    info_qvel_ = []
    info_mean_ = []
    info_logstd_ = []

    ravg = []
    
    for n in range(num_trajs):
        e.reset()
        returns = 0
        for t in range(e._max_episode_steps):
            obs = e.get_obs()
            obs_.append(obs)
            info_qpos_.append(e.env.data.qpos.ravel().copy())
            info_qvel_.append(e.env.data.qvel.ravel().copy())
            action, infos = pi.get_action(obs)
            action = pi.get_action(obs)[0] # eval
            
            if clip:
                action = np.clip(action, -1, 1)

            act_.append(action)
            info_mean_.append(infos['mean'])
            info_logstd_.append(infos['log_std'])

            _, rew, _, info = e.step(action)
            returns += rew
            rew_.append(rew)

            done = False
            if t == (e._max_episode_steps-1):
                done = True
            term_.append(done)

            #e.env.mj_render() # this is much faster
            # e.render()
        ravg.append(returns)
        print(n, returns, np.mean(ravg))

    # write out hdf5 file
    obs_ = np.array(obs_).astype(np.float32)
    act_ = np.array(act_).astype(np.float32)
    rew_ = np.array(rew_).astype(np.float32)
    term_ = np.array(term_).astype(np.bool_)
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
    dataset.create_dataset('infos/qpos', data=info_qpos_, compression='gzip')
    dataset.create_dataset('infos/qvel', data=info_qvel_, compression='gzip')
    dataset.create_dataset('infos/action_mean', data=info_mean_, compression='gzip')
    dataset.create_dataset('infos/action_logstd', data=info_logstd_, compression='gzip')

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

