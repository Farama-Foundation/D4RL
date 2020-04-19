"""
A quick script to run a sanity check on all environments.
"""
import gym
import d4rl
from gym.wrappers import TimeLimit

ENVS = [
    'minigrid-fourrooms-v0',
    'minigrid-fourrooms-random-v0',
]

if __name__ == '__main__':
    for env_name in ENVS:
        print('Checking', env_name)
        env = gym.make(env_name)
        assert type(env) == TimeLimit
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

        if env.observation_space.shape is not None:
            assert dset['observations'].shape[1:] == env.observation_space.shape, 'Observation shape does not match env shape: %s vs %s' % (str(dset['observations'].shape[1:]), str(env.observation_space.shape))
        assert dset['actions'].shape[1:] == env.action_space.shape, 'Action shape does not match env shape: %s vs %s' % (str(dset['actions'].shape[1:]), str(env.action_space.shape))
        score = env.get_normalized_score(0.0)

