import d4rl
import click 
import os
import gym
import numpy as np
import pickle
import h5py
import collections
from mjrl.utils.gym_env import GymEnv

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', default='door-v0')
def main(env_name):
    if env_name is "":
        print("Unknown env.")
        return
    demos = pickle.load(open('./demonstrations/'+env_name+'_demos.pickle', 'rb'))
    # render demonstrations
    demo_playback(env_name, demos, clip=True)

def demo_playback(env_name, demo_paths, clip=False):
    e = gym.make(env_name)
    e.reset()

    obs_ = []
    act_ = []
    rew_ = []
    term_ = []
    timeout_ = []
    info_qpos_ = []
    info_qvel_ = []
    info_env_state_ = collections.defaultdict(list)
    
    for i, path in enumerate(demo_paths):
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        returns = 0
        for t in range(actions.shape[0]):
            obs_.append(e.get_obs())
            info_qpos_.append(e.env.data.qpos.ravel().copy())
            info_qvel_.append(e.env.data.qvel.ravel().copy())
            [info_env_state_[k].append(v) for k,v in e.get_env_state().items()]
            commanded_action = actions[t]
            if clip:
                commanded_action = np.clip(commanded_action, -1.0, 1.0)
            act_.append(commanded_action)

            _, rew, _, info = e.step(commanded_action)
            returns += rew

            rew_.append(rew)

            done = False
            timeout = False
            if t == (actions.shape[0]-1):
                timeout = True
            #if t == (e._max_episode_steps-1):
            #    timeout = True
            #    done = False

            term_.append(done)
            timeout_.append(timeout)

            #e.env.mj_render() # this is much faster
            #e.render()
        print(i, returns, returns/float(actions.shape[0]))

    # write out hdf5 file
    obs_ = np.array(obs_).astype(np.float32)
    act_ = np.array(act_).astype(np.float32)
    rew_ = np.array(rew_).astype(np.float32)
    term_ = np.array(term_).astype(np.bool_)
    timeout_ = np.array(timeout_).astype(np.bool_)
    info_qpos_ = np.array(info_qpos_).astype(np.float32)
    info_qvel_ = np.array(info_qvel_).astype(np.float32)

    if clip:
        dataset = h5py.File('%s_demos_clipped.hdf5' % env_name, 'w')
    else:
        dataset = h5py.File('%s_demos.hdf5' % env_name, 'w')
    #dataset.create_dataset('observations', obs_.shape, dtype='f4')
    dataset.create_dataset('observations', data=obs_, compression='gzip')
    dataset.create_dataset('actions', data=act_, compression='gzip')
    dataset.create_dataset('rewards', data=rew_, compression='gzip')
    dataset.create_dataset('terminals', data=term_, compression='gzip')
    dataset.create_dataset('timeouts', data=timeout_, compression='gzip')
    #dataset['infos/qpos'] = info_qpos_
    #dataset['infos/qvel'] = info_qvel_
    for k in info_env_state_:
        dataset.create_dataset('infos/%s' % k, data=np.array(info_env_state_[k], dtype=np.float32), compression='gzip')

if __name__ == '__main__':
    main()
