import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer


class PointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.agent_bid = 0
        self.target_sid = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'point_mass.xml', 5)
        self.agent_bid = self.sim.model.body_name2id('agent')
        self.target_sid = self.sim.model.site_name2id('target')

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self.get_obs()
        reward = self.get_reward(obs)
        return obs, reward, False, dict(solved=(reward > -0.1), state=self.get_env_state())

    def get_obs(self):
        agent_pos = self.data.body_xpos[self.agent_bid].ravel()
        target_pos = self.data.site_xpos[self.target_sid].ravel()
        return np.concatenate([agent_pos[:2], self.data.qvel.ravel(), target_pos[:2]])

    def get_reward(self, obs, act=None):
        if len(obs.shape) == 1:
            # vector obs, called when stepping the env
            agent_pos = obs[:2]
            target_pos = obs[-2:]
            l1_dist = np.sum(np.abs(agent_pos - target_pos))
            l2_dist = np.linalg.norm(agent_pos - target_pos)
        else:
            obs = np.expand_dims(obs, axis=0) if len(obs.shape) == 2 else obs
            agent_pos = obs[:, :, :2]
            target_pos = obs[:, :, -2:]
            l1_dist = np.sum(np.abs(agent_pos - target_pos), axis=-1)
            l2_dist = np.linalg.norm(agent_pos - target_pos, axis=-1)
        reward = -1.0 * l1_dist - 0.5 * l2_dist
        return reward

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        rewards = self.get_reward(obs)
        rewards[..., :-1] = rewards[..., 1:]   # shift index by 1 to have r(s,a)=r(s')
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
        return paths

    def reset_model(self):
        # randomize the agent and goal
        agent_x = self.np_random.uniform(low=-1.0, high=1.0)
        agent_y = self.np_random.uniform(low=-1.0, high=1.0)
        goal_x  = self.np_random.uniform(low=-1.0, high=1.0)
        goal_y  = self.np_random.uniform(low=-1.0, high=1.0)
        qp = np.array([agent_x, agent_y])
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.site_pos[self.target_sid][0] = goal_x
        self.model.site_pos[self.target_sid][1] = goal_y
        self.sim.forward()
        return self.get_obs()

    def evaluate_success(self, paths, logger=None):
        success = 0.0
        for p in paths:
            if np.mean(p['env_infos']['solved'][-4:]) > 0.0:
                success += 1.0
        success_rate = 100.0*success/len(paths)
        if logger is None:
            # nowhere to log so return the value
            return success_rate
        else:
            # log the success
            # can log multiple statistics here if needed
            logger.log_kv('success_rate', success_rate)
            return None

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
        self.set_state(qp, qv)
        self.model.site_pos[self.target_sid] = target_pos
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.sim.forward()
