"""
A quick script to run a sanity check on all environments.
"""
import gym
import d4rl
import numpy as np

ENVS = [
    'halfcheetah-random-v0',
    'halfcheetah-medium-v0',
    'halfcheetah-expert-v0',
    'halfcheetah-mixed-v0',
    'halfcheetah-medium-expert-v0',
    'walker2d-random-v0',
    'walker2d-medium-v0',
    'walker2d-expert-v0',
    'walker2d-mixed-v0',
    'walker2d-medium-expert-v0',
    'hopper-random-v0',
    'hopper-medium-v0',
    'hopper-expert-v0',
    'hopper-mixed-v0',
    'hopper-medium-expert-v0',
    'maze2d-open-v0',
    'maze2d-umaze-v1',
    'maze2d-medium-v1',
    'maze2d-large-v1',
    'maze2d-open-dense-v0',
    'maze2d-umaze-dense-v1',
    'maze2d-medium-dense-v1',
    'maze2d-large-dense-v1',
    'minigrid-fourrooms-v0',
    'minigrid-fourrooms-random-v0',
    'pen-human-v0',
    'pen-cloned-v0',
    'pen-expert-v0',
    'hammer-human-v0',
    'hammer-cloned-v0',
    'hammer-expert-v0',
    'relocate-human-v0',
    'relocate-cloned-v0',
    'relocate-expert-v0',
    'door-human-v0',
    'door-cloned-v0',
    'door-expert-v0',
    'antmaze-umaze-v0',
    'antmaze-umaze-diverse-v0',
    'antmaze-medium-play-v0',
    'antmaze-medium-diverse-v0',
    'antmaze-large-play-v0',
    'antmaze-large-diverse-v0',
    'mini-kitchen-microwave-kettle-light-slider-v0',
    'kitchen-microwave-kettle-light-slider-v0',
    'kitchen-microwave-kettle-bottomburner-light-v0',
]

if __name__ == '__main__':
    for env_name in ENVS:
        print('Checking', env_name)
        try:
            env = gym.make(env_name)
        except:
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

        env.reset()
        env.step(env.action_space.sample())
        score = env.get_normalized_score(0.0)

        dset = d4rl.qlearning_dataset(env, dataset=dset)
        assert 'observations' in dset, 'Observations not in dataset'
        assert 'next_observations' in dset, 'Observations not in dataset'
        assert 'actions' in dset, 'Actions not in dataset'
        assert 'rewards' in dset, 'Rewards not in dataset'
        assert 'terminals' in dset, 'Terminals not in dataset'
        N = dset['observations'].shape[0]
        print('\t %d samples' % N)
        assert dset['next_observations'].shape[0] == N, 'NextObs number does not match (%d vs %d)' % (dset['actions'].shape[0], N)
        assert dset['actions'].shape[0] == N, 'Action number does not match (%d vs %d)' % (dset['actions'].shape[0], N)
        assert dset['rewards'].shape[0] == N, 'Reward number does not match (%d vs %d)' % (dset['rewards'].shape[0], N)
        assert dset['terminals'].shape[0] == N, 'Terminals number does not match (%d vs %d)' % (dset['terminals'].shape[0], N)
        print('\t num terminals: %d' % np.sum(dset['terminals']))
