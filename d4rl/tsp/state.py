from typing import NamedTuple, Union, Tuple

import torch
import numpy as np


class StateTSP(NamedTuple):
    # Fixed input
    loc: np.ndarray  # (n_loc, 2)
    dist: np.ndarray  # (n_loc, n_loc)

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    # ids: np.ndarray  # Keeps track of original fixed data index of rows

    # State
    first_a: np.ndarray  # int
    prev_a: np.ndarray  # int
    visited_: np.ndarray  # (n_loc,)Keeps track of nodes that have been visited
    lengths: np.ndarray  # int
    cur_coord: np.ndarray  # (2,)
    i: np.ndarray  # int # Keeps track of step

    @property
    def visited(self):
        return self.visited_

    def __getitem__(self, key: Union[np.ndarray, slice]):
        return self._replace(
            # ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )


    @classmethod
    def initialize(cls, loc, visited_dtype=np.uint8) -> "StateTSP":
        # loc: (n, 2)
        assert len(loc.shape) == 2, loc.shape

        n_loc, _ = loc.shape
        loc_tensor = torch.from_numpy(loc).float()
        state_tsp = cls(
            loc=loc,
            # dist=np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=-1),  # (n_loc, n_loc)
            dist=(loc_tensor[:, None, :] - loc_tensor[None, :, :]).norm(p=2, dim=-1).numpy(),
            first_a=0,
            prev_a=0,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                np.zeros(n_loc, dtype=np.uint8)
                if visited_dtype == np.uint8
                else np.zeros((n_loc + 63) // 64, dtype=np.long)  # Ceil
            ),
            lengths=0,
            cur_coord=None,
            i=0
        )

        return state_tsp

    def get_final_cost(self) -> float:
        # assert self.finished()
        return self.lengths + np.linalg.norm(self.loc[self.first_a] - self.cur_coord)

    def update(self, selected: int = None, move: int = None):
        if move is not None:
            # selected = self.get_nn_current().gather(-1, move.view(-1,1,1)).view(-1)
            selected = self.get_nn_current()

        # Update the state
        prev_a = selected  # Add dimension for step
        cur_coord = self.loc[prev_a].copy()

        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + np.linalg.norm(cur_coord - self.cur_coord)  # (batch_dim, 1)
        else:
            lengths = self.lengths

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i == 0 else self.first_a
        assert self.visited_[prev_a] == 0, (prev_a, self.visited_)
        self.visited_[prev_a] = 1

        return self._replace(
            first_a=first_a,
            prev_a=prev_a,
            visited_=self.visited_.copy(),
            lengths=lengths,
            cur_coord=cur_coord,
            i=self.i + 1
        )

    def finished(self):
        return self.i >= len(self.loc)

    def get_cur_node(self) -> np.ndarray:
        return self.loc[self.prev_a]

    # def get_mask(self) -> bool:
    #     return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        """Mask visited node, and return top K nodes for each unvisited node."""

        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = len(self.loc) - self.i  # Number of remaining

        
        full_neighbor_dist = self.dist + self.visited[:, None, :].astype(np.float32) * 1e6
        return np.argpartition(full_neighbor_dist, min(k, len(self.loc) - 1), -1)[:, :k]

    def get_nn_current(self, k=None) -> Tuple[np.ndarray, np.ndarray]:
        """Return k indexes related to the top-k nearlest neighbors"""

        if k is None:
            k = len(self.loc)
        k = min(k, len(self.loc))

        neighbor_dist = self.dist[self.prev_a]
        # mask neighbor_dist
        # import pdb; pdb.set_trace()
        masked_neighbor_dist = neighbor_dist + self.visited.astype(np.float32) * 1e6
        masked_neighbor_dist = torch.from_numpy(masked_neighbor_dist)
        neighbor_nodes = masked_neighbor_dist.topk(k, dim=-1, largest=False)[1].numpy()

        top_k_dist = masked_neighbor_dist[neighbor_nodes]
        neighbor_mask = np.ones(k)
        neighbor_mask[top_k_dist >= 1e6] = 0.
        # then recover maksed neighbor dist to -1
        masked_neighbor_dist[masked_neighbor_dist >= 1e6] = -1.

        assert len(neighbor_nodes) > 0

        return {
            "timestep": np.asarray([self.i]),
            "masked_neighbor_dist": masked_neighbor_dist,
            "neighbor_nodes": neighbor_nodes,
            "neighbor_mask": neighbor_mask,
            "global_mask": self.visited_.copy().astype(np.float32)
        }

    def get_full_nn_current(self, k=None):
        if k is None:
            k = len(self.loc)
        k = min(k, len(self.loc))  # Number of remaining

        neighbor_dist = self.dist[self.prev_a]
        # mask neighbor_dist
        # import pdb; pdb.set_trace()
        neighbor_dist = neighbor_dist + self.visited.astype(np.float32) * 1e6
        return neighbor_dist

    def construct_solutions(self, actions):
        return actions
