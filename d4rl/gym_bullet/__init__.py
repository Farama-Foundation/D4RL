from gym.envs.registration import register
from d4rl.gym_bullet import gym_envs

HOPPER_RANDOM_SCORE = -20.272305
HALFCHEETAH_RANDOM_SCORE = -280.178953
WALKER_RANDOM_SCORE = 1.629008
ANT_RANDOM_SCORE = -325.6

HOPPER_EXPERT_SCORE = 3234.3
HALFCHEETAH_EXPERT_SCORE = 12135.0
WALKER_EXPERT_SCORE = 4592.3
ANT_EXPERT_SCORE = 3879.7

register(
    id='bullet-hopper-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_hopper_env',
    max_episode_steps=1000,
)

register(
    id='bullet-halfcheetah-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
)

register(
    id='bullet-ant-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_ant_env',
    max_episode_steps=1000,
)

register(
    id='bullet-walker2d-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_walker_env',
    max_episode_steps=1000,
)

# Single Policy datasets
register(
    id='bullet-hopper-medium-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/hopper_medium.hdf5'
    }
)

register(
    id='bullet-halfcheetah-medium-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/halfcheetah_medium.hdf5'
    }
)

register(
    id='bullet-walker2d-medium-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/walker2d_medium.hdf5'
    }
)

register(
    id='bullet-hopper-expert-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/hopper_expert.hdf5'
    }
)

register(
    id='bullet-halfcheetah-expert-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/halfcheetah_expert.hdf5'
    }
)

register(
    id='bullet-walker2d-expert-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/walker2d_expert.hdf5'
    }
)

register(
    id='bullet-hopper-random-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/hopper_random.hdf5'
    }
)

register(
    id='bullet-halfcheetah-random-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/halfcheetah_random.hdf5'
    }
)

register(
    id='bullet-walker2d-random-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/walker2d_random.hdf5'
    }
)

# Mixed datasets
register(
    id='bullet-hopper-medium-replay-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/hopper_mixed.hdf5'
    },
)

register(
    id='bullet-walker2d-medium-replay-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/walker_mixed.hdf5'
    }
)

register(
    id='bullet-halfcheetah-medium-replay-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/halfcheetah_mixed.hdf5'
    }
)

# Mixtures of random/medium and experts
register(
    id='bullet-walker2d-medium-expert-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/walker2d_medium_expert.hdf5'
    }
)

register(
    id='bullet-halfcheetah-medium-expert-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/halfcheetah_medium_expert.hdf5'
    }
)

register(
    id='bullet-hopper-medium-expert-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/hopper_medium_expert.hdf5'
    }
)

register(
    id='bullet-ant-medium-expert-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_ant_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': ANT_RANDOM_SCORE,
        'ref_max_score': ANT_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/ant_medium_expert.hdf5'
    }
)

register(
    id='bullet-ant-medium-replay-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_ant_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': ANT_RANDOM_SCORE,
        'ref_max_score': ANT_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/ant_mixed.hdf5'
    }
)

register(
    id='bullet-ant-medium-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_ant_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': ANT_RANDOM_SCORE,
        'ref_max_score': ANT_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/ant_medium.hdf5'
    }
)

register(
    id='bullet-ant-random-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_ant_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': ANT_RANDOM_SCORE,
        'ref_max_score': ANT_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/ant_random.hdf5'
    }
)

register(
    id='bullet-ant-expert-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_ant_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': ANT_RANDOM_SCORE,
        'ref_max_score': ANT_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/ant_expert.hdf5'
    }
)

register(
    id='bullet-ant-random-expert-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_ant_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': ANT_RANDOM_SCORE,
        'ref_max_score': ANT_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_bullet/ant_random_expert.hdf5'
    }
)
