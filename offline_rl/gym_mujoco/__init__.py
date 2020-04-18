from gym.envs.registration import register
from offline_rl.gym_mujoco import gym_envs

HOPPER_RANDOM_SCORE = -20.272305
HALFCHEETAH_RANDOM_SCORE = -280.178953
WALKER_RANDOM_SCORE = 1.629008

HOPPER_EXPERT_SCORE = 3234.3
HALFCHEETAH_EXPERT_SCORE = 12135.0
WALKER_EXPERT_SCORE = 4592.3

# Single Policy datasets
register(
    id='hopper-medium-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5'
    }
)

register(
    id='halfcheetah-medium-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5'
    }
)

register(
    id='walker2d-medium-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium.hdf5'
    }
)

register(
    id='hopper-expert-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5'
    }
)

register(
    id='halfcheetah-expert-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_expert.hdf5'
    }
)

register(
    id='walker2d-expert-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_expert.hdf5'
    }
)

register(
    id='hopper-random-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5'
    }
)

register(
    id='halfcheetah-random-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random.hdf5'
    }
)

register(
    id='walker2d-random-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random.hdf5'
    }
)

# Mixed datasets
register(
    id='hopper-mixed-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_mixed.hdf5'
    },
)

register(
    id='walker2d-mixed-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker_mixed.hdf5'
    }
)

register(
    id='halfcheetah-mixed-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_mixed.hdf5'
    }
)

# Mixtures of random/medium and experts
register(
    id='walker2d-random-expert-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random_expert.hdf5'
    }
)

register(
    id='halfcheetah-random-expert-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random_expert.hdf5'
    }
)

register(
    id='walker2d-medium-expert-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_walker_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': WALKER_RANDOM_SCORE,
        'ref_max_score': WALKER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium_expert.hdf5'
    }
)

register(
    id='halfcheetah-medium-expert-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HALFCHEETAH_RANDOM_SCORE,
        'ref_max_score': HALFCHEETAH_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium_expert.hdf5'
    }
)

register(
    id='hopper-medium-expert-v0',
    entry_point='offline_rl.gym_mujoco.gym_envs:get_hopper_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': HOPPER_RANDOM_SCORE,
        'ref_max_score': HOPPER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium_expert.hdf5'
    }
)
