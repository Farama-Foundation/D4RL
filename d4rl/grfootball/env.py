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
        args: Namespace,
        stacked: bool = False,
        render: bool = False,
        channel_dimensions: Tuple[int, int] = (42, 42),
        debug: bool = False,
        reward_type: str = "basic",
        encoder_type: str = "basic",
        representation: str = "raw",
    ) -> None:
        """Create a Google Research Football environment.

        Parameters
        ----------
        scenario_id : str
            The scenario id, refer to ...
        n_right_agents : int
            Number of moving player at right side.
        n_left_agents : int
            Number of moving player at left side.
        stacked : bool, optional
            Stack observation or not, by default False
        render : bool, optional
            Enable render or not, by default False
        channel_dimensions : Tuple[int, int], optional
            Channel dimensions specification, by default (42, 42)
        debug : bool, optional
            Enable debug mode or not
        reward_type: str, optional
            Reward func type, by default basic
        encoder_type: str, optional
            Encoder type, by default basic
        use_builtin_gk: bool, optional
            Indicate use builtin goalkeepers or not, by default True
        """

        super().__init__()

        self.debug = debug

        scenario_id = args.map_name
        n_right_players = args.n_right_players
        n_left_players = args.n_left_players
        use_builtin_gk = args.use_builtin_gk

        self.n_players = n_right_players + n_left_players
        self.n_agents = self.n_players
        self.n_left_players = n_left_players
        self.n_right_players = n_right_players
        self.n_left_control = n_left_players
        self.n_right_control = n_right_players

        if use_builtin_gk and n_left_players > 0:
            self.n_left_control -= 1
            self.n_agents -= 1

        if use_builtin_gk and n_right_players > 0:
            self.n_right_control -= 1
            self.n_agents -= 1

        self.env = football_env.create_environment(
            env_name=scenario_id,
            stacked=stacked,
            representation=representation,
            logdir=os.path.join(tempfile.gettempdir(), "grfootball"),
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=render,
            dump_frequency=0,
            number_of_left_players_agent_controls=self.n_left_control,
            number_of_right_players_agent_controls=self.n_right_control,
            channel_dimensions=channel_dimensions,
        )
        self.env_core = retrieve_env_core(self.env)
        self.reward_func = get_reward_func(reward_type)()
        self.num_actions = 19
        self.encoder = get_encoder(encoder_type)(
            self.n_left_control, self.n_right_players, self.num_actions
        )
        self.representation = representation
        self.use_builtin_gk = use_builtin_gk

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=self.encoder.shape, dtype=np.float32
        )
        self.share_observation_space = spaces.Box(low=-10.0, high=10.0, shape=self.get_state([self.observation_space.sample()] * self.n_agents)[0].shape, dtype=np.float32)

        # last frame
        self.last_frame: Frame = None

    def _build_observation_from_raw(self) -> List[np.ndarray]:
        """
        get the observation of all player's in teams
        """

        def encode_obs(raw_obs):
            obs = self.encoder.encode(raw_obs)

            obs_cat = np.hstack(
                [np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)]
            )

            return obs_cat, obs["avail"]

        raw_obs_list = self.env.observation()

        # if enable builtin goal keeper, we need to pop its observation
        if self.use_builtin_gk and len(raw_obs_list) == self.n_players:
            raw_obs_list = raw_obs_list[1:self.n_left_players] + raw_obs_list[self.n_left_players + 1:]
        obs_list, ava_action_list = list(
            zip(*[encode_obs(r_obs) for r_obs in raw_obs_list])
        )
        return obs_list, ava_action_list

    def get_obs(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return a list of observation for all moving agents.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple of observation list and action_mask list.
        """

        if self.representation == "raw":
            obs, action_masks = self._build_observation_from_raw()
        else:
            raise NotImplementedError("only raw representation is supported.")
            # assert not self.use_builtin_gk
            # obs = self.env.observation()
            # # all one
            # action_masks = [np.ones(self.action_space.n) for _ in range(self.n_agents)]
        return obs, action_masks

    def step(self, actions: Union[Dict[AgentID, int], Sequence[int]]) -> Tuple:
        """A single environment step. Returns a tuple of ...

        Parameters
        ----------
        actions : Union[Dict[AgentID, int], Sequence[int]]
            A dict or sequence of agents' actions. If a sequence, the actions are sorted from left to right.

        Returns
        -------
        Tuple
            A tuple of (observations, states, rewards, dones, infos, available_actions), all of them are `np.ndarray`, except info.
        """

        if isinstance(actions, Dict):
            actions = [int(v) for k, v in sorted(actions)]
        elif isinstance(actions, Sequence):
            actions = list(map(int, actions))
        elif isinstance(actions, np.ndarray) and len(actions.shape) > 1:
            actions = actions.squeeze()

        # if self.use_builtin_gk and self.n_left_control > 0:
        #     actions.insert(0, 19)
        # if self.use_builtin_gk and self.n_right_control > 0:
        #     actions.insert(self.n_left_players, 19)

        if self.debug:
            logging.debug("Actions".center(60, "-"))

        raw_observations, raw_rewards, done, info = self.env.step(actions)
        raw_observations = copy.deepcopy(raw_observations)

        # reward shaping
        rewards = np.asarray(
            [
                self.reward_func.calc_reward(_r, _prev_obs, _obs)
                for _r, _prev_obs, _obs in zip(
                    raw_rewards.tolist(), self.last_frame.observations, raw_observations
                )
            ],
            dtype=np.float32,
        )

        observations, available_actions = list(map(np.asarray, self.get_obs()))
        dones = np.asarray([done] * len(observations), dtype=bool)
        infos = [info.copy() for _ in range(len(observations))]
        states = self.get_state(observations)

        # update last frame
        self.last_frame = Frame(
            actions=actions,
            rewards=raw_rewards,
            observations=raw_observations,
            states=states,
        )

        # expand rewards
        rewards = np.expand_dims(rewards, -1)

        # local observation, global state, rewards, done, info, action masks
        return observations, states, rewards, dones, infos, available_actions

    def get_state(self, observations: Sequence[np.ndarray]) -> np.ndarray:
        """Retrieve state for each agent and transform them into numpy array-like data.

        Parameters
        ----------
        obserations : Sequence[np.ndarray]
            A sequence of parsed observation

        Returns
        -------
        List[np.ndarray]
            A stacked state, where the first dim is equal to self.n_agents
        """

        # concat all observations by group
        repeats = self.n_agents
        assert len(observations) == repeats, (len(observations), repeats)

        state = np.concatenate(observations).reshape(1, -1)

        states = np.tile(state, (repeats, 1))

        return states

    def reset(self) -> Tuple:
        """Reset environment.

        Returns
        -------
        Tuple
            A tuple of observations, states and available actions.
        """

        raw_observations = self.env.reset()
        observations, available_actions = list(map(np.asarray, self.get_obs()))
        states = np.asarray(self.get_state(observations))
        self.last_frame = Frame(
            actions=np.zeros(len(states), dtype=int),
            rewards=np.zeros(len(states), dtype=np.float32),
            observations=raw_observations,
            states=states,
        )
        return observations, states, available_actions

    def render(self):
        return super().render()

    def close(self):
        self.env.close()

    def save_replay(self):
        raise NotImplementedError
        # self.env.write_dump("shutdown")


if __name__ == "__main__":
    import time

    args = Namespace(map_name="5_vs_5", n_right_agents=5, n_left_agents=5)
    env = GRFootball(args)

    observations, states, available_actions = env.reset()
    n_agents = env.n_agents
    act_space = env.action_space

    def compute_action(
        observation: List[np.ndarray], available_actions: List[np.ndarray]
    ):
        assert len(observation) == n_agents
        act = []
        for ava_actions in available_actions:
            idxes = np.where(ava_actions == 1)[0]
            act.append(np.random.choice(idxes))
        return act

    try:
        start = time.time()
        n_frame = 0

        while True:
            actions = compute_action(observations, available_actions)
            observations, states, rewards, dones, infos, available_actions = env.step(
                actions
            )
            n_frame += 1
            if n_frame % 10 == 0:
                cur_time = time.time()
                print(
                    "FPS: {:.3} reward: {:.3f} {:.3f} {:.3f} done: {}".format(
                        n_frame / (cur_time - start),
                        np.mean(rewards),
                        np.max(rewards),
                        np.min(rewards),
                        any(dones),
                    )
                )
            if any(dones):
                break
    except KeyboardInterrupt:
        env.close()