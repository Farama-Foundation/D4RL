from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv
from d4rl.hand_manipulation_suite.door_v0 import DoorEnvV0
from d4rl.hand_manipulation_suite.hammer_v0 import HammerEnvV0
from d4rl.hand_manipulation_suite.pen_v0 import PenEnvV0
from d4rl.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

DOOR_RANDOM_SCORE = -56.512833
DOOR_EXPERT_SCORE = 2880.5693087298737

HAMMER_RANDOM_SCORE = -274.856578
HAMMER_EXPERT_SCORE = 12794.134825156867

PEN_RANDOM_SCORE = 96.262799
PEN_EXPERT_SCORE = 3076.8331017826877

RELOCATE_RANDOM_SCORE = -6.425911
RELOCATE_EXPERT_SCORE = 4233.877797728884

# Swing the door open
register(
    id='door-v0',
    entry_point='d4rl.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
)

register(
    id='door-human-v0',
    entry_point='d4rl.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': DOOR_RANDOM_SCORE,
        'ref_max_score': DOOR_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_demos_clipped.hdf5'
    }
)

register(
    id='door-cloned-v0',
    entry_point='d4rl.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': DOOR_RANDOM_SCORE,
        'ref_max_score': DOOR_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-demos-v0-bc-combined.hdf5'
    }
)

register(
    id='door-expert-v0',
    entry_point='d4rl.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': DOOR_RANDOM_SCORE,
        'ref_max_score': DOOR_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_expert_clipped.hdf5'
    }
)

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='d4rl.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)

register(
    id='hammer-human-v0',
    entry_point='d4rl.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': HAMMER_RANDOM_SCORE,
        'ref_max_score': HAMMER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_demos_clipped.hdf5'
    }
)

register(
    id='hammer-cloned-v0',
    entry_point='d4rl.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': HAMMER_RANDOM_SCORE,
        'ref_max_score': HAMMER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-demos-v0-bc-combined.hdf5'
    }
)

register(
    id='hammer-expert-v0',
    entry_point='d4rl.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': HAMMER_RANDOM_SCORE,
        'ref_max_score': HAMMER_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_expert_clipped.hdf5'
    }
)


# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='d4rl.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
)

register(
    id='pen-human-v0',
    entry_point='d4rl.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': PEN_RANDOM_SCORE,
        'ref_max_score': PEN_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_demos_clipped.hdf5'
    }
)

register(
    id='pen-cloned-v0',
    entry_point='d4rl.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': PEN_RANDOM_SCORE,
        'ref_max_score': PEN_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-demos-v0-bc-combined.hdf5'
    }
)

register(
    id='pen-expert-v0',
    entry_point='d4rl.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
    kwargs={
        'ref_min_score': PEN_RANDOM_SCORE,
        'ref_max_score': PEN_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_expert_clipped.hdf5'
    }
)


# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='d4rl.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)

register(
    id='relocate-human-v0',
    entry_point='d4rl.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': RELOCATE_RANDOM_SCORE,
        'ref_max_score': RELOCATE_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_demos_clipped.hdf5'
    }
)

register(
    id='relocate-cloned-v0',
    entry_point='d4rl.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': RELOCATE_RANDOM_SCORE,
        'ref_max_score': RELOCATE_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-demos-v0-bc-combined.hdf5'
    }
)

register(
    id='relocate-expert-v0',
    entry_point='d4rl.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
    kwargs={
        'ref_min_score': RELOCATE_RANDOM_SCORE,
        'ref_max_score': RELOCATE_EXPERT_SCORE,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_expert_clipped.hdf5'
    }
)

