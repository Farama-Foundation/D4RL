import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer


class Reacher7DOFEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.hand_sid = -2
        self.target_sid = -1
        mujoco_env.MujocoEnv.__init__(self, 'sawyer.xml', 4)
        utils.EzPickle.__init__(self)
        self.hand_sid = self.model.site_name2id("finger")
        self.target_sid = self.model.site_name2id("target")

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self.get_obs()
        reward = self.get_reward(obs, a)
        return obs, reward, False, self.get_env_infos()

    def get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.ravel() * self.dt,       # delta_x instead of velocity
            self.data.site_xpos[self.hand_sid],
            self.data.site_xpos[self.target_sid],
        ])

    def get_reward(self, obs, act=None):
        obs = np.clip(obs, -10.0, 10.0)
        if len(obs.shape) == 1:
            # vector obs, called when stepping the env
            hand_pos = obs[-6:-3]
            target_pos = obs[-3:]
            l1_dist = np.sum(np.abs(hand_pos - target_pos))
            l2_dist = np.linalg.norm(hand_pos - target_pos)
        else:
            obs = np.expand_dims(obs, axis=0) if len(obs.shape) == 2 else obs
            hand_pos = obs[:, :, -6:-3]
            target_pos = obs[:, :, -3:]
            l1_dist = np.sum(np.abs(hand_pos - target_pos), axis=-1)
            l2_dist = np.linalg.norm(hand_pos - target_pos, axis=-1)
        reward = - l1_dist - 5.0 * l2_dist
        return reward

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        rewards = self.get_reward(obs)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self):
        target_pos = np.array([0.1, 0.1, 0.1])
        target_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
        target_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
        target_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        self.model.site_pos[self.target_sid] = target_pos
        self.sim.forward()

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        self.target_reset()
        return self.get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.site_pos[self.target_sid].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        self.model.site_pos[self.target_sid] = target_pos
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 2.0
