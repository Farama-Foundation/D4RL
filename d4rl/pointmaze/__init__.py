from gym.envs.registration import register

from .maze_model import (
    LARGE_MAZE,
    LARGE_MAZE_EVAL,
    MEDIUM_MAZE,
    MEDIUM_MAZE_EVAL,
    OPEN,
    U_MAZE,
    U_MAZE_EVAL,
    MazeEnv,
)

register(
    id="maze2d-open-v0",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=150,
    kwargs={
        "maze_spec": OPEN,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 0.01,
        "ref_max_score": 20.66,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-sparse.hdf5",
    },
)

register(
    id="maze2d-umaze-v0",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=150,
    kwargs={
        "maze_spec": U_MAZE,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 0.94,
        "ref_max_score": 62.6,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse.hdf5",
    },
)

register(
    id="maze2d-medium-v0",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=250,
    kwargs={
        "maze_spec": MEDIUM_MAZE,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 5.77,
        "ref_max_score": 85.14,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse.hdf5",
    },
)


register(
    id="maze2d-large-v0",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=600,
    kwargs={
        "maze_spec": LARGE_MAZE,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 4.83,
        "ref_max_score": 191.99,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse.hdf5",
    },
)


register(
    id="maze2d-umaze-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_spec": U_MAZE,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 23.85,
        "ref_max_score": 161.86,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5",
    },
)

register(
    id="maze2d-medium-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=600,
    kwargs={
        "maze_spec": MEDIUM_MAZE,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 13.13,
        "ref_max_score": 277.39,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5",
    },
)


register(
    id="maze2d-large-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=800,
    kwargs={
        "maze_spec": LARGE_MAZE,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 6.7,
        "ref_max_score": 273.99,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5",
    },
)

register(
    id="maze2d-eval-umaze-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_spec": U_MAZE_EVAL,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 36.63,
        "ref_max_score": 141.4,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-umaze-sparse-v1.hdf5",
    },
)

register(
    id="maze2d-eval-medium-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=600,
    kwargs={
        "maze_spec": MEDIUM_MAZE_EVAL,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 13.07,
        "ref_max_score": 204.93,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-medium-sparse-v1.hdf5",
    },
)


register(
    id="maze2d-eval-large-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=800,
    kwargs={
        "maze_spec": LARGE_MAZE_EVAL,
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 16.4,
        "ref_max_score": 302.22,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-large-sparse-v1.hdf5",
    },
)


register(
    id="maze2d-open-dense-v0",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=150,
    kwargs={
        "maze_spec": OPEN,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 11.17817,
        "ref_max_score": 27.166538620695782,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-dense.hdf5",
    },
)

register(
    id="maze2d-umaze-dense-v0",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=150,
    kwargs={
        "maze_spec": U_MAZE,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 23.249793,
        "ref_max_score": 81.78995240126592,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-dense.hdf5",
    },
)

register(
    id="maze2d-medium-dense-v0",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=250,
    kwargs={
        "maze_spec": MEDIUM_MAZE,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 19.477620,
        "ref_max_score": 96.03474232952358,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-dense.hdf5",
    },
)


register(
    id="maze2d-large-dense-v0",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=600,
    kwargs={
        "maze_spec": LARGE_MAZE,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 27.388310,
        "ref_max_score": 215.09965671563742,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-dense.hdf5",
    },
)

register(
    id="maze2d-umaze-dense-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_spec": U_MAZE,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 68.537689,
        "ref_max_score": 193.66285642381482,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-dense-v1.hdf5",
    },
)

register(
    id="maze2d-medium-dense-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=600,
    kwargs={
        "maze_spec": MEDIUM_MAZE,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 44.264742,
        "ref_max_score": 297.4552547777125,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-dense-v1.hdf5",
    },
)


register(
    id="maze2d-large-dense-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=800,
    kwargs={
        "maze_spec": LARGE_MAZE,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 30.569041,
        "ref_max_score": 303.4857382709002,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-dense-v1.hdf5",
    },
)

register(
    id="maze2d-eval-umaze-dense-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=300,
    kwargs={
        "maze_spec": U_MAZE_EVAL,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 56.95455,
        "ref_max_score": 178.21373133248397,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-umaze-dense-v1.hdf5",
    },
)

register(
    id="maze2d-eval-medium-dense-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=600,
    kwargs={
        "maze_spec": MEDIUM_MAZE_EVAL,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 42.28578,
        "ref_max_score": 235.5658957482388,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-medium-dense-v1.hdf5",
    },
)


register(
    id="maze2d-eval-large-dense-v1",
    entry_point="d4rl.pointmaze:MazeEnv",
    max_episode_steps=800,
    kwargs={
        "maze_spec": LARGE_MAZE_EVAL,
        "reward_type": "dense",
        "reset_target": False,
        "ref_min_score": 56.95455,
        "ref_max_score": 326.09647655082637,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-large-dense-v1.hdf5",
    },
)
