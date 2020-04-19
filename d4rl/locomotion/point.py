# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wrapper for creating the point environment."""

import math
import numpy as np
import mujoco_py
import os

from gym import utils
from gym.envs.mujoco import mujoco_env
from d4rl.locomotion import mujoco_goal_env

from d4rl.locomotion import goal_reaching_env
from d4rl.locomotion import maze_env

MY_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'assets')


class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  FILE = os.path.join(MY_ASSETS_DIR, 'point.xml')

  def __init__(self, file_path=None, expose_all_qpos=False):
    if file_path is None:
        file_path = self.FILE

    self._expose_all_qpos = expose_all_qpos

    mujoco_env.MujocoEnv.__init__(self, file_path, 1)
    # mujoco_goal_env.MujocoGoalEnv.__init__(self, file_path, 1)
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

  def step(self, action):
    action[0] = 0.2 * action[0]
    qpos = np.copy(self.physics.data.qpos)
    qpos[2] += action[1]
    ori = qpos[2]
    # Compute increment in each direction.
    dx = math.cos(ori) * action[0]
    dy = math.sin(ori) * action[0]
    # Ensure that the robot is within reasonable range.
    qpos[0] = np.clip(qpos[0] + dx, -100, 100)
    qpos[1] = np.clip(qpos[1] + dy, -100, 100)
    qvel = self.physics.data.qvel
    self.set_state(qpos, qvel)
    for _ in range(0, self.frame_skip):
      self.physics.step()
    next_obs = self._get_obs()
    reward = 0
    done = False
    info = {}
    return next_obs, reward, done, info

  def _get_obs(self):
    if self._expose_all_qpos:
      return np.concatenate([
          self.physics.data.qpos.flat[:3],  # Only point-relevant coords.
          self.physics.data.qvel.flat[:3]])
    return np.concatenate([
        self.physics.data.qpos.flat[2:3],
        self.physics.data.qvel.flat[:3]])

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(
        size=self.physics.model.nq, low=-.1, high=.1)
    qvel = self.init_qvel + self.np_random.randn(self.physics.model.nv) * .1

    # Set everything other than point to original position and 0 velocity.
    qpos[3:] = self.init_qpos[3:]
    qvel[3:] = 0.
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


class GoalReachingPointEnv(goal_reaching_env.GoalReachingEnv, PointEnv):
  """Point locomotion rewarded for goal-reaching."""
  BASE_ENV = PointEnv

  def __init__(self, goal_sampler=goal_reaching_env.disk_goal_sampler,
               file_path=None,
               expose_all_qpos=False):
    goal_reaching_env.GoalReachingEnv.__init__(self, goal_sampler)
    PointEnv.__init__(self,
                      file_path=file_path,
                      expose_all_qpos=expose_all_qpos)

class GoalReachingPointDictEnv(goal_reaching_env.GoalReachingDictEnv, PointEnv):
  """Ant locomotion for goal reaching in a disctionary compatible format."""
  BASE_ENV = PointEnv

  def __init__(self, goal_sampler=goal_reaching_env.disk_goal_sampler,
               file_path=None,
               expose_all_qpos=False):
    goal_reaching_env.GoalReachingDictEnv.__init__(self, goal_sampler)
    PointEnv.__init__(self, 
                    file_path=file_path,
                    expose_all_qpos=expose_all_qpos)

class PointMazeEnv(maze_env.MazeEnv, GoalReachingPointEnv):
  """Point navigating a maze."""
  LOCOMOTION_ENV = GoalReachingPointEnv

  def __init__(self, goal_sampler=None, expose_all_qpos=True,
               *args, **kwargs):
    if goal_sampler is None:
      goal_sampler = lambda np_rand: maze_env.MazeEnv.goal_sampler(self, np_rand)
    maze_env.MazeEnv.__init__(
        self, *args, manual_collision=True,
        goal_sampler=goal_sampler,
        expose_all_qpos=expose_all_qpos,
        **kwargs)


def create_goal_reaching_policy(obs_to_goal=lambda obs: obs[-2:],
                                obs_to_ori=lambda obs: obs[0]):
  """A hard-coded policy for reaching a goal position."""

  def policy_fn(obs):
    goal_x, goal_y = obs_to_goal(obs)
    goal_dist = np.linalg.norm([goal_x, goal_y])
    goal_ori = np.arctan2(goal_y, goal_x)
    ori = obs_to_ori(obs)
    ori_diff = (goal_ori - ori) % (2 * np.pi)

    radius = goal_dist / 2. / max(0.1, np.abs(np.sin(ori_diff)))
    rotation_left = (2 * ori_diff) % np.pi
    circumference_left = max(goal_dist, radius * rotation_left)

    speed = min(circumference_left * 5., 1.0)
    velocity = speed
    if ori_diff > np.pi / 2 and ori_diff < 3 * np.pi / 2:
      velocity *= -1

    time_left = min(circumference_left / (speed * 0.2), 10.)
    signed_ori_diff = ori_diff
    if signed_ori_diff >= 3 * np.pi / 2:
      signed_ori_diff = 2 * np.pi - signed_ori_diff
    elif signed_ori_diff > np.pi / 2 and signed_ori_diff < 3 * np.pi / 2:
      signed_ori_diff = signed_ori_diff - np.pi

    angular_velocity = signed_ori_diff / time_left
    angular_velocity = np.clip(angular_velocity, -1., 1.)

    return np.array([velocity, angular_velocity])

  return policy_fn


def create_maze_navigation_policy(maze_env):
  """Creates a hard-coded policy to navigate a maze."""
  ori_index = 2 if maze_env._expose_all_qpos else 0
  obs_to_ori = lambda obs: obs[ori_index]

  goal_reaching_policy = create_goal_reaching_policy(obs_to_ori=obs_to_ori)
  goal_reaching_policy_fn = lambda obs, goal: goal_reaching_policy(
    np.concatenate([obs, goal]))

  return maze_env.create_navigation_policy(goal_reaching_policy_fn)
