"""
A quick script to run a sanity check on all environments.
"""
import gym
import d4rl
import numpy as np

ENVS = [
    'bullet-halfcheetah-random-v0',
    'bullet-halfcheetah-medium-v0',
    'bullet-halfcheetah-expert-v0',
    'bullet-halfcheetah-medium-replay-v0',
    'bullet-halfcheetah-medium-expert-v0',
    'bullet-walker2d-random-v0',
    'bullet-walker2d-medium-v0',
    'bullet-walker2d-expert-v0',
    'bullet-walker2d-medium-replay-v0',
    'bullet-walker2d-medium-expert-v0',
    'bullet-hopper-random-v0',
    'bullet-hopper-medium-v0',
    'bullet-hopper-expert-v0',
    'bullet-hopper-medium-replay-v0',
    'bullet-hopper-medium-expert-v0',
    'bullet-ant-random-v0',
    'bullet-ant-medium-v0',
    'bullet-ant-expert-v0',
    'bullet-ant-medium-replay-v0',
    'bullet-ant-medium-expert-v0',
    'bullet-maze2d-open-v0',
    'bullet-maze2d-umaze-v0',
    'bullet-maze2d-medium-v0',
    'bullet-maze2d-large-v0',
]

if __name__ == '__main__':
    for env_name in ENVS:
        print('Checking', env_name)
        try:
            env = gym.make(env_name)
        except Exception as e:
            print(e)
            continue
        dset = env.get_dataset()
        print('\t Max episode steps:', env._max_episode_steps)
        print('\t',dset['observations'].shape, dset['actions'].shape)
        assert 'observations' in dset, 'Observations not in dataset'
        assert 'actions' in dset, 'Actions not in dataset'
        assert 'rewards' in dset, 'Rewards not in dataset'
        assert 'terminals' in dset, 'Terminals not in dataset'
        N = dset['observations'].shape[0]
        print('\t %d samples' % N)
        assert dset['actions'].shape[0] == N, 'Action number does not match (%d vs %d)' % (dset['actions'].shape[0], N)
        assert dset['rewards'].shape[0] == N, 'Reward number does not match (%d vs %d)' % (dset['rewards'].shape[0], N)
        assert dset['terminals'].shape[0] == N, 'Terminals number does not match (%d vs %d)' % (dset['terminals'].shape[0], N)
        print('\t num terminals: %d' % np.sum(dset['terminals']))
        print('\t avg rew: %f' % np.mean(dset['rewards']))

        env.reset()
        env.step(env.action_space.sample())
        score = env.get_normalized_score(0.0)

