"""Wrapper for creating the swimmer environment."""

import math
import numpy as np
import mujoco_py
import os

from gym import utils
from gym.envs.mujoco import mujoco_env
from d4rl.locomotion import mujoco_goal_env

from d4rl.locomotion import goal_reaching_env
from d4rl.locomotion import maze_env
from d4rl import offline_env

GYM_ASSETS_DIR = os.path.join(
    os.path.dirname(mujoco_env.__file__),
    'assets')


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  """Basic swimmer locomotion environment."""
  FILE = os.path.join(GYM_ASSETS_DIR, 'swimmer.xml')

  def __init__(self, file_path=None, expose_all_qpos=False, non_zero_reset=False):
    if file_path is None:
      file_path = self.FILE

    self._expose_all_qpos = expose_all_qpos

    mujoco_env.MujocoEnv.__init__(self, file_path, 5)
    utils.EzPickle.__init__(self)

  @property
  def physics(self):
    # Check mujoco version is greater than version 1.50 to call correct physics
    # model containing PyMjData object for getting and setting position/velocity.
    # Check https://github.com/openai/mujoco-py/issues/80 for updates to api.
    if mujoco_py.get_version() >= '1.50':
      return self.sim
    else:
      return self.model

  def _step(self, a):
    return self.step(a)

  def step(self, a):
    ctrl_cost_coeff = 0.0001
    xposbefore = self.sim.data.qpos[0]
    self.do_simulation(a, self.frame_skip)
    xposafter = self.sim.data.qpos[0]
    reward_fwd = (xposafter - xposbefore) / self.dt
    reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
    reward = reward_fwd + reward_ctrl
    ob = self._get_obs()
    return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

  def _get_obs(self):
    if self._expose_all_qpos:
      obs = np.concatenate([
          self.physics.data.qpos.flat[:5],  # Ensures only swimmer obs.
          self.physics.data.qvel.flat[:5],
      ])
    else:
      obs = np.concatenate([
          self.physics.data.qpos.flat[2:5],
          self.physics.data.qvel.flat[:5],
      ])

    return obs

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(
        size=self.model.nq, low=-.1, high=.1)
    qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

    # Set everything other than swimmer to original position and 0 velocity.
    qpos[5:] = self.init_qpos[5:]
    qvel[5:] = 0.
    self.set_state(qpos, qvel)
    return self._get_obs()

  def get_xy(self):
    return self.physics.data.qpos[:2]

  def set_xy(self, xy):
    qpos = np.copy(self.physics.data.qpos)
    qpos[0] = xy[0]
    qpos[1] = xy[1]
    qvel = self.physics.data.qvel
    self.set_state(qpos, qvel)


class GoalReachingSwimmerEnv(goal_reaching_env.GoalReachingEnv, SwimmerEnv):
  """Swimmer locomotion rewarded for goal-reaching."""
  BASE_ENV = SwimmerEnv

  def __init__(self, goal_sampler=goal_reaching_env.disk_goal_sampler,
               file_path=None,
               expose_all_qpos=False, non_zero_reset=False, eval=False, reward_type="dense", **kwargs):
    goal_reaching_env.GoalReachingEnv.__init__(self, goal_sampler, eval=eval, reward_type=reward_type)
    SwimmerEnv.__init__(self,
                        file_path=file_path,
                        expose_all_qpos=expose_all_qpos, 
                        non_zero_reset=non_zero_reset)

class SwimmerMazeEnv(maze_env.MazeEnv, GoalReachingSwimmerEnv, offline_env.OfflineEnv):
  """Swimmer navigating a maze."""
  LOCOMOTION_ENV = GoalReachingSwimmerEnv

  def __init__(self, goal_sampler=None, expose_all_qpos=True,
               reward_type='dense',
               *args, **kwargs):
    if goal_sampler is None:
      goal_sampler = lambda np_rand: maze_env.MazeEnv.goal_sampler(self, np_rand)
    maze_env.MazeEnv.__init__(
        self, *args, manual_collision=False,
        goal_sampler=goal_sampler,
        expose_all_qpos=expose_all_qpos,
        reward_type=reward_type,
        **kwargs)
    offline_env.OfflineEnv.__init__(self, **kwargs)
    
  def set_target(self, target_location=None):
    return self.set_target_goal(target_location) 
