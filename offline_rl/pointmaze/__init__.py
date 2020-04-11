from .maze_model import MazeEnv, OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE
from gym.envs.registration import register

register(
    id='maze2d-open-v0',
    entry_point='offline_rl.pointmaze:MazeEnv',
    max_episode_steps=50,
    kwargs={
        'maze_spec':OPEN,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 0.01,
        'ref_max_score': 20.66,
        'dataset_url':'gs://justinjfu-public/maze2d/maze2d-open-sparse.hdf5'
    }
)

register(
    id='maze2d-umaze-v0',
    entry_point='offline_rl.pointmaze:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 0.94,
        'ref_max_score': 62.6,
        'dataset_url':'gs://justinjfu-public/maze2d/maze2d-umaze-sparse.hdf5'
    }
)

register(
    id='maze2d-medium-v0',
    entry_point='offline_rl.pointmaze:MazeEnv',
    max_episode_steps=250,
    kwargs={
        'maze_spec':MEDIUM_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 5.77,
        'ref_max_score': 85.14,
        'dataset_url':'gs://justinjfu-public/maze2d/maze2d-medium-sparse.hdf5'
    }
)


register(
    id='maze2d-large-v0',
    entry_point='offline_rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':LARGE_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'gs://justinjfu-public/maze2d/maze2d-large-sparse.hdf5'
    }
)


register(
    id='maze2d-open-dense-v0',
    entry_point='offline_rl.pointmaze:MazeEnv',
    max_episode_steps=50,
    kwargs={
        'maze_spec':OPEN,
        'reward_type':'dense',
        'reset_target': False,
        'ref_min_score': 11.17817,
        'ref_max_score': 27.166538620695782,
        'dataset_url':'gs://justinjfu-public/maze2d/maze2d-open-dense.hdf5'
    }
)

register(
    id='maze2d-umaze-dense-v0',
    entry_point='offline_rl.pointmaze:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'dense',
        'reset_target': False,
        'ref_min_score': 23.249793,
        'ref_max_score': 81.78995240126592,
        'dataset_url':'gs://justinjfu-public/maze2d/maze2d-umaze-dense.hdf5'
    }
)

register(
    id='maze2d-medium-dense-v0',
    entry_point='offline_rl.pointmaze:MazeEnv',
    max_episode_steps=250,
    kwargs={
        'maze_spec':MEDIUM_MAZE,
        'reward_type':'dense',
        'reset_target': False,
        'ref_min_score': 19.477620,
        'ref_max_score': 96.03474232952358,
        'dataset_url':'gs://justinjfu-public/maze2d/maze2d-medium-dense.hdf5'
    }
)


register(
    id='maze2d-large-dense-v0',
    entry_point='offline_rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':LARGE_MAZE,
        'reward_type':'dense',
        'reset_target': False,
        'ref_min_score': 27.388310,
        'ref_max_score': 215.09965671563742,
        'dataset_url':'gs://justinjfu-public/maze2d/maze2d-large-dense.hdf5'
    }
)

