def run_policy_on_env(policy_fn, env, truncate_episode_at=None, first_obs=None):
    if first_obs is None:
        obs = env.reset()
    else:
        obs = first_obs

    trajectory = []
    step_num = 0
    while True:
        act = policy_fn(obs)
        next_obs, rew, done, _ = env.step(act)
        trajectory.append((obs, act, rew, done))
        obs = next_obs
        step_num += 1
        if done or (
            truncate_episode_at is not None and step_num >= truncate_episode_at
        ):
            break
    return trajectory
