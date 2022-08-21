from argparse import Namespace
from typing import List, Any, Tuple, Dict, Union

import os
import logging

import gym
import numpy as np
import pickle

from gym import spaces

from d4rl.offline_env import OfflineEnv

from .tsp import TSP  # type: ignore pylance
from .state import StateTSP  # type: ignore pylance


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class TSPEnv(gym.Env):

    def __init__(self, scale: int, action_dim: int, dataset_path: str, node_embedding_path: str, use_raw_state: bool = False) -> None:
        super().__init__()

        self.scale = scale
        self.dataset_path = dataset_path
        self.node_embedding_path = node_embedding_path
        self.cur_task = None
        self.cur_state: StateTSP = None
        self.dataset = None
        self.sequence_idx = None
        self.cur_ava_nodes = None
        self.cur_node_mask = None
        self.node_embeddings = None
        self.ctx_embedding = None

        # coord dim + action_mask_length
        self.action_dim = action_dim
        # 1: time_step, 128: coord_embedding, action_dim: action_mask, 100/200: neighbor_distance, 
        self.state_dim = 1 + 128 + scale # self.action_dim # + scale

        self._action_space = spaces.Discrete(self.scale)

        if use_raw_state:
            self._observation_space = spaces.Dict({
                "time_step": spaces.Box(low=0, high=1., shape=(128,), dtype=np.float32),
                "coord_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32),
                "neighbor_mask": spaces.Box(low=0., high=1., shape=(self.action_dim,), dtype=np.float32),
                # "neighbor_distance": spaces.Box(low=-1e6, high=1e6, shape=(self.action_dim,), dtype=np.float32)
            })
        else:
            self._observation_space = spaces.Box(low=-1e6, high=1e6, shape=(self.state_dim,), dtype=np.float32)

        self.use_raw_state = use_raw_state

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def preprocess_obs(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        obs = raw_obs["obs"]
        assert obs.shape[-1] == 2, obs.shape

        action_mask = raw_obs["coord_distance"]
        assert action_mask.shape[-1] == self.scale, action_mask

        obs_np = np.concatenate([obs, action_mask], axis=-1)
        assert obs_np.shape[-1] == self.state_dim, obs_np.shape

        return obs_np

    def get_cur_action_mask(self) -> np.ndarray:
        return self.cur_node_mask.copy()

    def get_cur_ava_nodes(self) -> np.ndarray:
        return self.cur_ava_nodes.copy()

    def step(self, node_idx: int):
        old_length = self.cur_state.lengths
        choose_node = self.cur_ava_nodes[node_idx]
        state_np = self.step_obs(choose_node)
        reward = -(self.cur_state.lengths - old_length)
        done = self.cur_state.finished()

        info = {
            "gap": (self.cur_state.get_final_cost() / self.dataset.val[self.sequence_idx] - 1) * 100,
            "cost": self.cur_state.get_final_cost(),
            "val": self.dataset.val[self.sequence_idx]
        }

        return state_np, reward, done, info

    def step_obs(self, choose_node: Union[int, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.cur_state is None:
            # init with a sequence of node
            self.cur_state = TSP.make_state(choose_node)
            start_move = self.dataset.soln[self.sequence_idx][0]
            self.cur_state = self.cur_state.update(start_move)
        else:
            self.cur_state = self.cur_state.update(choose_node)

        state = self.cur_state

        idx = state.prev_a
        node_embedding = self.node_embeddings[self.sequence_idx][idx] # + self.ctx_embedding
        raw_state_np: Dict[str, np.ndarray] = state.get_nn_current()
        # import pdb; pdb.set_trace()

        # TODO(ming): neighbor mask or global mask?
        #   global mask contains the trajectory information
        if self.use_raw_state:
            state_np = {
                "time_step": raw_state_np["timestep"] / self.scale,
                "coord_embedding": node_embedding,
                # "global_mask": raw_state_np["global_mask"],
                # "neighbor_mask": raw_state_np["neighbor_mask"],
                "neighbor_distance": raw_state_np["maksed_neighbor_dist"]
            }
        else:
            flattens = [
                raw_state_np["timestep"] / self.scale,
                node_embedding,
                # raw_state_np["global_mask"],
                raw_state_np["masked_neighbor_dist"]
            ]
            state_np = np.concatenate(flattens, axis=-1)

        # update neighbor nodes, and mask
        # first top cur_nodes
        self.cur_ava_nodes = raw_state_np["neighbor_nodes"][:self.action_dim]
        self.cur_node_mask = raw_state_np["neighbor_mask"][:self.action_dim]

        return state_np

    def reset(self, traj_idx: int = None) -> Any:
        if self.dataset is None:
            with open(self.dataset_path, 'rb') as f:
                data_all = pickle.load(f)
                self.dataset = Namespace(**{
                    "coords": np.asarray(data_all["data"]),
                    "soln": np.asarray(data_all["seq"]),
                    "val": np.asarray(data_all["val"]),
                    "time": np.asarray(data_all["time"]),
                })
            self.node_embeddings = np.load(self.node_embedding_path)

        size = self.dataset.coords.shape[0]

        # random select a start point as state
        if traj_idx is not None:
            self.sequence_idx = traj_idx
        else:
            self.sequence_idx = np.random.choice(size)
        self.ctx_embedding = np.mean(self.node_embeddings[self.sequence_idx], axis=0)
        self.cur_state = None
        self.cur_ava_nodes = None
        self.cur_node_mask = None
        # load from local storage
        sequence = self.dataset.coords[self.sequence_idx]
        state_np = self.step_obs(sequence)

        return state_np
        
    def seed(self, seed: int = ...) -> List[int]:
        np.random.RandomState(seed)


class OfflineTSPEnv(TSPEnv, OfflineEnv):

    def __init__(self, **kwargs) -> None:
        TSPEnv.__init__(self, kwargs['scale'], kwargs['action_dim'], kwargs['env_dataset_path'], kwargs['env_node_embedding_path'])
        OfflineEnv.__init__(self, **kwargs)

    def get_dataset(self, h5path=None):
        dataset = super().get_dataset(h5path)
        dataset["actions"] = dataset["actions"].astype(np.int32)
        return dataset

    def get_normalized_score(self, score):
        self.ref_max_score = -self.dataset.val[self.sequence_idx]
        if self.ref_min_score is None:
            raise ValueError("Reference score not provided for env")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)
