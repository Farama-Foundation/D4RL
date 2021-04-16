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

"""Adapted from efficient-hrl maze_env.py."""

import os
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np
import gym
from copy import deepcopy

RESET = R = 'r'  # Reset position.
GOAL = G = 'g'

# Maze specifications for dataset generation
U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, 0, 0, 1],
          [1, 1, 1, 0, 1],
          [1, G, 0, 0, 1],
          [1, 1, 1, 1, 1]]

BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, G, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, G, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, G, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 1, G, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, G, 0, 1, 0, 0, G, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, G, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, G, 0, G, 1, 0, G, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# Maze specifications with a single target goal
U_MAZE_TEST = [[1, 1, 1, 1, 1],
              [1, R, 0, 0, 1],
              [1, 1, 1, 0, 1],
              [1, G, 0, 0, 1],
              [1, 1, 1, 1, 1]]

BIG_MAZE_TEST = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE_TEST = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# Maze specifications for evaluation
U_MAZE_EVAL = [[1, 1, 1, 1, 1],
              [1, 0, 0, R, 1],
              [1, 0, 1, 1, 1],
              [1, 0, 0, G, 1],
              [1, 1, 1, 1, 1]]

BIG_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 0, G, 1],
                [1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 1],
                [1, 1, 1, 0, 0, 1, 1, 1],
                [1, G, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, G, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, R, 0, 1, G, 0, 0, 1, 0, G, 0, 1],
                    [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, G, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                    [1, G, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 0, G, 1, G, 0, 0, 0, G, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

U_MAZE_EVAL_TEST = [[1, 1, 1, 1, 1],
              [1, 0, 0, R, 1],
              [1, 0, 1, 1, 1],
              [1, 0, 0, G, 1],
              [1, 1, 1, 1, 1]]

BIG_MAZE_EVAL_TEST = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 0, G, 1],
                [1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 1],
                [1, 1, 1, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE_EVAL_TEST = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, R, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
                    [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


class MazeEnv(gym.Env):
  LOCOMOTION_ENV = None  # Must be specified by child class.

  def __init__(
      self,
      maze_map,
      maze_size_scaling,
      maze_height=0.5,
      manual_collision=False,
      non_zero_reset=False,
      reward_type='dense',
      *args,
      **kwargs):
    if self.LOCOMOTION_ENV is None:
      raise ValueError('LOCOMOTION_ENV is unspecified.')

    xml_path = self.LOCOMOTION_ENV.FILE
    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    self._maze_map = maze_map

    self._maze_height = maze_height
    self._maze_size_scaling = maze_size_scaling
    self._manual_collision = manual_collision

    self._maze_map = maze_map

    # Obtain a numpy array form for a maze map in case we want to reset
    # to multiple starting states
    temp_maze_map = deepcopy(self._maze_map)
    for i in range(len(maze_map)):
      for j in range(len(maze_map[0])):
        if temp_maze_map[i][j] in [RESET,]:
          temp_maze_map[i][j] = 0
        elif temp_maze_map[i][j] in [GOAL,]:
          temp_maze_map[i][j] = 1
    
    self._np_maze_map = np.array(temp_maze_map)

    torso_x, torso_y = self._find_robot()
    self._init_torso_x = torso_x
    self._init_torso_y = torso_y

    for i in range(len(self._maze_map)):
      for j in range(len(self._maze_map[0])):
        struct = self._maze_map[i][j]
        if struct == 1:  # Unmovable block.
          # Offset all coordinates so that robot starts at the origin.
          ET.SubElement(
              worldbody, "geom",
              name="block_%d_%d" % (i, j),
              pos="%f %f %f" % (j * self._maze_size_scaling - torso_x,
                                i * self._maze_size_scaling - torso_y,
                                self._maze_height / 2 * self._maze_size_scaling),
              size="%f %f %f" % (0.5 * self._maze_size_scaling,
                                 0.5 * self._maze_size_scaling,
                                 self._maze_height / 2 * self._maze_size_scaling),
              type="box",
              material="",
              contype="1",
              conaffinity="1",
              rgba="0.7 0.5 0.3 1.0",
          )

    torso = tree.find(".//body[@name='torso']")
    geoms = torso.findall(".//geom")

    _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
    tree.write(file_path)

    self.LOCOMOTION_ENV.__init__(self, *args, file_path=file_path, non_zero_reset=non_zero_reset, reward_type=reward_type, **kwargs)

    self.target_goal = None

  def _xy_to_rowcol(self, xy):
    size_scaling = self._maze_size_scaling
    xy = (max(xy[0], 1e-4), max(xy[1], 1e-4))
    return (int(1 + (xy[1]) / size_scaling),
            int(1 + (xy[0]) / size_scaling))
  
  def _get_reset_location(self,):
    prob = (1.0 - self._np_maze_map) / np.sum(1.0 - self._np_maze_map) 
    prob_row = np.sum(prob, 1)
    row_sample = np.random.choice(np.arange(self._np_maze_map.shape[0]), p=prob_row)
    col_sample = np.random.choice(np.arange(self._np_maze_map.shape[1]), p=prob[row_sample] * 1.0 / prob_row[row_sample])
    reset_location = self._rowcol_to_xy((row_sample, col_sample))
    
    # Add some random noise
    random_x = np.random.uniform(low=0, high=0.5) * 0.5 * self._maze_size_scaling
    random_y = np.random.uniform(low=0, high=0.5) * 0.5 * self._maze_size_scaling

    return (max(reset_location[0] + random_x, 0), max(reset_location[1] + random_y, 0))

  def _rowcol_to_xy(self, rowcol, add_random_noise=False):
    row, col = rowcol
    x = col * self._maze_size_scaling - self._init_torso_x
    y = row * self._maze_size_scaling - self._init_torso_y
    if add_random_noise:
      x = x + np.random.uniform(low=0, high=self._maze_size_scaling * 0.25)
      y = y + np.random.uniform(low=0, high=self._maze_size_scaling * 0.25)
    return (x, y)

  def goal_sampler(self, np_random, only_free_cells=True, interpolate=True):
    valid_cells = []
    goal_cells = []

    for i in range(len(self._maze_map)):
      for j in range(len(self._maze_map[0])):
        if self._maze_map[i][j] in [0, RESET, GOAL] or not only_free_cells:
          valid_cells.append((i, j))
        if self._maze_map[i][j] == GOAL:
          goal_cells.append((i, j))

    # If there is a 'goal' designated, use that. Otherwise, any valid cell can
    # be a goal.
    sample_choices = goal_cells if goal_cells else valid_cells
    cell = sample_choices[np_random.choice(len(sample_choices))]
    xy = self._rowcol_to_xy(cell, add_random_noise=True)

    random_x = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling
    random_y = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling

    xy = (max(xy[0] + random_x, 0), max(xy[1] + random_y, 0))

    return xy
  
  def set_target_goal(self, goal_input=None):
    if goal_input is None:
      self.target_goal = self.goal_sampler(np.random)
    else:
      self.target_goal = goal_input
    
    print ('Target Goal: ', self.target_goal)
    ## Make sure that the goal used in self._goal is also reset:
    self._goal = self.target_goal

  def _find_robot(self):
    structure = self._maze_map
    size_scaling = self._maze_size_scaling
    for i in range(len(structure)):
      for j in range(len(structure[0])):
        if structure[i][j] == RESET:
          return j * size_scaling, i * size_scaling
    raise ValueError('No robot in maze specification.')

  def _is_in_collision(self, pos):
    x, y = pos
    structure = self._maze_map
    size_scaling = self._maze_size_scaling
    for i in range(len(structure)):
      for j in range(len(structure[0])):
        if structure[i][j] == 1:
          minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
          maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
          miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
          maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
          if minx <= x <= maxx and miny <= y <= maxy:
            return True
    return False

  def step(self, action):
    if self._manual_collision:
      old_pos = self.get_xy()
      inner_next_obs, inner_reward, done, info = self.LOCOMOTION_ENV.step(self, action)
      new_pos = self.get_xy()
      if self._is_in_collision(new_pos):
        self.set_xy(old_pos)
    else:
      inner_next_obs, inner_reward, done, info = self.LOCOMOTION_ENV.step(self, action)
    next_obs = self._get_obs()
    return next_obs, inner_reward, done, info

  def _get_best_next_rowcol(self, current_rowcol, target_rowcol):
    """Runs BFS to find shortest path to target and returns best next rowcol. 
       Add obstacle avoidance"""
    current_rowcol = tuple(current_rowcol)
    target_rowcol = tuple(target_rowcol)
    if target_rowcol == current_rowcol:
        return target_rowcol

    visited = {}
    to_visit = [target_rowcol]
    while to_visit:
      next_visit = []
      for rowcol in to_visit:
        visited[rowcol] = True
        row, col = rowcol
        left = (row, col - 1)
        right = (row, col + 1)
        down = (row + 1, col)
        up = (row - 1, col)
        for next_rowcol in [left, right, down, up]:
          if next_rowcol == current_rowcol:  # Found a shortest path.
            return rowcol
          next_row, next_col = next_rowcol
          if next_row < 0 or next_row >= len(self._maze_map):
            continue
          if next_col < 0 or next_col >= len(self._maze_map[0]):
            continue
          if self._maze_map[next_row][next_col] not in [0, RESET, GOAL]:
            continue
          if next_rowcol in visited:
            continue
          next_visit.append(next_rowcol)
      to_visit = next_visit

    raise ValueError('No path found to target.')

  def create_navigation_policy(self,
                               goal_reaching_policy_fn,
                               obs_to_robot=lambda obs: obs[:2], 
                               obs_to_target=lambda obs: obs[-2:],
                               relative=False):
    """Creates a navigation policy by guiding a sub-policy to waypoints."""

    def policy_fn(obs):
      # import ipdb; ipdb.set_trace()
      robot_x, robot_y = obs_to_robot(obs)
      robot_row, robot_col = self._xy_to_rowcol([robot_x, robot_y])
      target_x, target_y = self.target_goal
      if relative:
        target_x += robot_x  # Target is given in relative coordinates.
        target_y += robot_y
      target_row, target_col = self._xy_to_rowcol([target_x, target_y])
      print ('Target: ', target_row, target_col, target_x, target_y)
      print ('Robot: ', robot_row, robot_col, robot_x, robot_y)

      waypoint_row, waypoint_col = self._get_best_next_rowcol(
          [robot_row, robot_col], [target_row, target_col])
      
      if waypoint_row == target_row and waypoint_col == target_col:
        waypoint_x = target_x
        waypoint_y = target_y
      else:
        waypoint_x, waypoint_y = self._rowcol_to_xy([waypoint_row, waypoint_col], add_random_noise=True)

      goal_x = waypoint_x - robot_x
      goal_y = waypoint_y - robot_y

      print ('Waypoint: ', waypoint_row, waypoint_col, waypoint_x, waypoint_y)

      return goal_reaching_policy_fn(obs, (goal_x, goal_y))

    return policy_fn
