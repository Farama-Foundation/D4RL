import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer

class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.qpos[0]
        
        delta = (xposafter - xposbefore)
        # make agent move in the negative x direction
        reward = -10.0 * delta
        done = False

        ob = self.get_obs()
        return ob, reward, done, self.get_env_infos()

    def get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos_init = self.init_qpos.copy()
        qpos_init[2] = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.set_state(qpos_init, self.init_qvel)
        self.sim.forward()
        return self.get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy())

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        self.set_state(qp, qv)
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
        self.viewer.cam.distance = self.model.stat.extent*1.2