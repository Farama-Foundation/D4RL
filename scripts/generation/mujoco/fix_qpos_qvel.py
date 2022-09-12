import argparse
import os

import gym
import h5py
import numpy as np

import d4rl
import d4rl.offline_env


def unwrap_env(env):
    return env.env.wrapped_env


def set_state_qpos(env, qpos, qvel):
    env.set_state(qpos, qvel)


def pad_obs(env, obs, twod=False, scale=0.1):
    # TODO: sample val
    if twod:
        val = env.init_qpos[0:2] + np.random.uniform(size=2, low=-0.1, high=0.1)
        state = np.concatenate([np.ones(2) * val, obs])
    else:
        val = env.init_qpos[0:1] + np.random.uniform(size=1, low=-scale, high=scale)
        state = np.concatenate([np.ones(1) * val, obs])
    return state


def set_state_obs(env, obs):
    env_name = str(unwrap_env(env).__class__)
    ant_env = "Ant" in env_name
    hopper_walker_env = "Hopper" in env_name or "Walker" in env_name
    state = pad_obs(env, obs, twod=ant_env, scale=0.005 if hopper_walker_env else 0.1)
    qpos_dim = env.sim.data.qpos.size
    if ant_env:
        env.set_state(state[:15], state[15:29])
    else:
        env.set_state(state[:qpos_dim], state[qpos_dim:])


def resync_state_obs(env, obs):
    # Prevents drifting of the obs over time
    ant_env = "Ant" in (str(unwrap_env(env).__class__))
    cur_qpos, cur_qvel = (
        env.sim.data.qpos.ravel().copy(),
        env.sim.data.qvel.ravel().copy(),
    )
    if ant_env:
        cur_qpos[2:15] = obs[0:13]
        cur_qvel = obs[13:27]
        env.set_state(cur_qpos, cur_qvel)
    else:
        qpos_dim = env.sim.data.qpos.size
        cur_qpos[1:] = obs[0 : qpos_dim - 1]
        cur_qvel = obs[qpos_dim - 1 :]
        env.set_state(cur_qpos, cur_qvel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    args = parser.parse_args()

    env = gym.make(args.env)
    env.reset()

    fname = unwrap_env(env).dataset_url.split("/")[-1]
    prefix, ext = os.path.splitext(fname)
    # out_fname = prefix+'_qfix'+ext
    out_fname = prefix + ext

    dset = env.get_dataset()
    all_qpos = dset["infos/qpos"]
    all_qvel = dset["infos/qvel"]
    observations = dset["observations"]
    actions = dset["actions"]
    dones = dset["terminals"]
    timeouts = dset["timeouts"]
    terminals = dones + timeouts

    start_obs = observations[0]
    set_state_obs(env, start_obs)
    # set_state_qpos(env, all_qpos[0], all_qvel[0])

    new_qpos = []
    new_qvel = []

    for t in range(actions.shape[0]):
        cur_qpos, cur_qvel = (
            env.sim.data.qpos.ravel().copy(),
            env.sim.data.qvel.ravel().copy(),
        )
        new_qpos.append(cur_qpos)
        new_qvel.append(cur_qvel)

        next_obs, reward, done, infos = env.step(actions[t])

        if t == actions.shape[0] - 1:
            break
        if terminals[t]:
            set_state_obs(env, observations[t + 1])
            # print(t, 'done')
        else:
            true_next_obs = observations[t + 1]
            error = ((true_next_obs - next_obs) ** 2).sum()
            if t % 1000 == 0:
                print(t, error)

            # prevent drifting over time
            resync_state_obs(env, observations[t + 1])

    dset_filepath = d4rl.offline_env.download_dataset_from_url(
        unwrap_env(env).dataset_url
    )
    inf = h5py.File(dset_filepath, "r")
    outf = h5py.File(out_fname, "w")

    for k in d4rl.offline_env.get_keys(inf):
        print("writing", k)
        if "qpos" in k:
            outf.create_dataset(k, data=np.array(new_qpos), compression="gzip")
        elif "qvel" in k:
            outf.create_dataset(k, data=np.array(new_qvel), compression="gzip")
        else:
            try:
                if "reward" in k:
                    outf.create_dataset(
                        k,
                        data=inf[k][:].squeeze().astype(np.float32),
                        compression="gzip",
                    )
                else:
                    if "terminals" in k or "timeouts" in k:
                        outf.create_dataset(
                            k, data=inf[k][:].astype(np.bool), compression="gzip"
                        )
                    else:
                        outf.create_dataset(
                            k, data=inf[k][:].astype(np.float32), compression="gzip"
                        )
            except Exception as e:
                print(e)
                outf.create_dataset(k, data=inf[k])
    outf.close()
