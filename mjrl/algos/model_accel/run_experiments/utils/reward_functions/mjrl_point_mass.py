import numpy as np

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_traj, horizon)
    obs = paths["observations"]
    obs = np.expand_dims(obs, axis=0) if len(obs.shape) == 2 else obs
    agent_pos = obs[:, :, :2]
    target_pos = obs[:, :, -2:]
    l1_dist = np.sum(np.abs(agent_pos - target_pos), axis=-1)
    l2_dist = np.linalg.norm(agent_pos - target_pos, axis=-1)
    rewards = -1.0 * l1_dist - 0.5 * l2_dist
    rewards[..., :-1] = rewards[..., 1:]   # shift index by 1 to have r(s,a)=r(s')
    paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
    return paths
