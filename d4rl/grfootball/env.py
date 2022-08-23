from argparse import Namespace
import os
import tempfile
import copy
import traceback

from typing import Tuple, Union, Dict, List, Sequence, Any
from collections import namedtuple

import numpy as np
import gym

from gym import spaces
from absl import logging

try:
    import gfootball

    from gfootball.env import football_env_core
    from gfootball.env.football_env_core import FootballEnvCore
    from gfootball import env as football_env
except ImportError as e:
    print(traceback.format_exc())
    raise e

from d4rl.grfootball.encoders import get_encoder
from d4rl.grfootball.reward_funcs import get_reward_func


AgentID = str
Frame = namedtuple("Frame", "actions,rewards,observations,states")


def retrieve_env_core(env) -> FootballEnvCore:
    if isinstance(env, (gym.ObservationWrapper, gym.Wrapper)):
        return retrieve_env_core(env.env)
    elif isinstance(env, football_env.football_env.FootballEnv):
        return env._env
    else:
        raise ValueError("unknow environment type: {}".format(type(env)))


class GRFootball(gym.Env):
    def __init__(
        self,
        scenario: str,
        n_agent: int,
        reward_type: str = "scoring"
    ) -> None:
        super().__init__()

        self.scenario = scenario
        self.n_agents = n_agent
        self.reward_type = reward_type

        self.env = football_env.create_environment(env_name=self.scenario,
                                                   number_of_left_players_agent_controls=self.n_agents,
                                                   representation="raw",
                                                   # representation="simple115v2",
                                                   rewards=self.reward_type)
        self.feature_encoder = get_encoder("basic")()
        self.reward_encoder = get_reward_func("basic")()

        self.action_space = [gym.spaces.Discrete(self.env.action_space.nvec[1]) for n in range(self.n_agents)]

        tmp_obs_dicts = self.env.reset()
        tmp_obs = [self._encode_obs(obs_dict)[0] for obs_dict in tmp_obs_dicts]
        self.observation_space = [spaces.Box(low=float("-inf"), high=float("inf"), shape=tmp_obs[n].shape, dtype=np.float32)
                                  for n in range(self.n_agents)]
        self.share_observation_space = self.observation_space.copy()

        self.pre_obs = None
        self._max_episode_steps = 3000

    def _build_observation_from_raw(self) -> List[np.ndarray]:
        """
        get the observation of all player's in teams
        """

    def _encode_obs(self, raw_obs):
            obs = self.feature_encoder.encode(raw_obs.copy())
            obs_cat = np.hstack(
                [np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)]
            )
            ava = obs["avail"]
            return obs_cat, ava

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        obs_dicts = self.env.reset()
        self.pre_obs = obs_dicts
        obs = []
        ava = []
        for obs_dict in obs_dicts:
            obs_i, ava_i = self._encode_obs(obs_dict)
            obs.append(obs_i)
            ava.append(ava_i)
        state = obs.copy()
        return np.asarray(obs, dtype=np.float32), np.asarray(state, dtype=np.float32), np.asarray(ava, dtype=np.float32)

    def step(self, actions):
        actions_int = [int(a) for a in actions]
        o, r, d, i = self.env.step(actions_int)
        obs = []
        ava = []
        for obs_dict in o:
            obs_i, ava_i = self._encode_obs(obs_dict)
            obs.append(obs_i)
            ava.append(ava_i)
        state = obs.copy()

        rewards = [[self.reward_encoder.calc_reward(_r, _prev_obs, _obs)]
                   for _r, _prev_obs, _obs in zip(r, self.pre_obs, o)]

        self.pre_obs = o

        dones = np.ones((self.n_agents), dtype=bool) * d
        infos = [i for n in range(self.n_agents)]
        return np.asarray(obs, dtype=np.float32), np.asarray(state, dtype=np.float32), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=np.bool_), infos, np.asarray(ava, dtype=np.float32)

    def render(self, **kwargs):
        # self.env.render(**kwargs)
        pass

    def close(self):
        pass

    def seed(self, args):
        pass

    def get_env_info(self):

        env_info = {"state_shape": self.observation_space[0].shape,
                    "obs_shape": self.observation_space[0].shape,
                    "n_actions": self.action_space[0].n,
                    "n_agents": self.n_agents,
                    "action_spaces": self.action_space
                    }
        return env_info


    def save_replay(self):
        raise NotImplementedError
        # self.env.write_dump("shutdown")
