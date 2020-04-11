"""
A quick script to run a sanity check on all environments.
"""
import gym
import offline_rl

ENVS = [
    'maze2d-open-v0',
    'maze2d-umaze-v0',
    'maze2d-medium-v0',
    'maze2d-large-v0',
    'maze2d-open-dense-v0',
    'maze2d-umaze-dense-v0',
    'maze2d-medium-dense-v0',
    'maze2d-large-dense-v0',
    'pen-human-v0',
    'pen-demos-v0',
    'pen-human-bc-combined-v0',
    'pen-expert-v0',
    'hammer-human-v0',
    'hammer-demos-v0',
    'hammer-human-bc-combined-v0',
    'hammer-expert-v0',
    'relocate-human-v0',
    'relocate-demos-v0',
    'relocate-human-bc-combined-v0',
    'relocate-expert-v0',
    'door-human-v0',
    'door-demos-v0',
    'door-human-bc-combined-v0',
    'door-expert-v0',
    'halfcheetah-random-v0',
    'halfcheetah-medium-v0',
    'halfcheetah-expert-v0',
    'halfcheetah-medium-expert-v0',
    'walker2d-random-v0',
    'walker2d-medium-v0',
    'walker2d-expert-v0',
    'walker2d-medium-expert-v0',
    'hopper-random-v0',
    'hopper-medium-v0',
    'hopper-expert-v0',
    'hopper-medium-expert-v0',
    'antmaze-umaze-noisy-v0',
    'antmaze-umaze-noisy-diverse-v0',
    'antmaze-medium-noisy-play-v0',
    'antmaze-medium-noisy-diverse-v0',
    'antmaze-large-noisy-play-v0',
    'antmaze-large-noisy-diverse-v0',
]

if __name__ == '__main__':
    for env_name in ENVS:
        print('Checking', env_name)
        env = gym.make(env_name)
        dset = env.get_dataset()
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

        assert dset['observations'].shape[1:] == env.observation_space.shape, 'Observation shape does not match env shape: %s vs %s' % (str(dset['observations'].shape[1:]), str(env.observation_space.shape))
        assert dset['actions'].shape[1:] == env.action_space.shape, 'Action shape does not match env shape: %s vs %s' % (str(dset['actions'].shape[1:]), str(env.action_space.shape))
        score = env.get_normalized_score(0.0)

