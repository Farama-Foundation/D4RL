from gym.envs.registration import register

register(
    id="minigrid-fourrooms-v0",
    entry_point="d4rl.gym_minigrid.envs.fourrooms:FourRoomsEnv",
    max_episode_steps=50,
    kwargs={
        "ref_min_score": 0.01442,
        "ref_max_score": 2.89685,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/minigrid/minigrid4rooms.hdf5",
    },
)

register(
    id="minigrid-fourrooms-random-v0",
    entry_point="d4rl.gym_minigrid.envs.fourrooms:FourRoomsEnv",
    max_episode_steps=50,
    kwargs={
        "ref_min_score": 0.01442,
        "ref_max_score": 2.89685,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/minigrid/minigrid4rooms_random.hdf5",
    },
)
