from typing import List, Type, Dict, Any

import h5py
import numpy as np
import pickle

class DatasetWriter(object):
    def __init__(self, mujoco=False, goal=False):
        self.mujoco = mujoco
        self.goal = goal
        self.data = self._reset_data()
        self._num_samples = 0

    def _reset_data(self):
        data = {
            'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
        }
        if self.mujoco:
            data['infos/qpos'] = []
            data['infos/qvel'] = []
        if self.goal:
            data['infos/goal'] = []
        return data

    def __len__(self):
        return self._num_samples

    def append_data(self, s, a, r, done, goal=None, mujoco_env_data=None):
        self._num_samples += 1
        self.data['observations'].append(s)
        self.data['actions'].append(a)
        self.data['rewards'].append(r)
        self.data['terminals'].append(done)
        if self.goal:
            self.data['infos/goal'].append(goal)
        if self.mujoco:
            self.data['infos/qpos'].append(mujoco_env_data.qpos.ravel().copy())
            self.data['infos/qvel'].append(mujoco_env_data.qvel.ravel().copy())

    def write_dataset(self, fname, max_size=None, compression='gzip'):
        np_data = {}
        for k in self.data:
            if k == 'terminals':
                dtype = np.bool_
            else:
                dtype = np.float32
            data = np.array(self.data[k], dtype=dtype)
            if max_size is not None:
                data = data[:max_size]
            np_data[k] = data

        dataset = h5py.File(fname, 'w')
        for k in np_data:
            dataset.create_dataset(k, data=np_data[k], compression=compression)
        dataset.close()


EpisodeID = str
AgentID = int
StepID = int

NDArray = Type[np.ndarray]


class Trajectory:
    def __init__(self, episode_id: str, agents: List[AgentID], max_episode_length: int, extra_keys: List[str] = []):
        self.episode_id = episode_id
        self.agents = agents
        self.max_episode_length = max_episode_length
        self.keys = ['observations', 'actions', 'terminals', 'rewards']
        self.keys.extend(extra_keys)
        self.extra_keys = extra_keys
        self.agent_trajectories = [[] for _ in agents]
        self.n_step = 0

    def __len__(self):
        return self.n_step

    def record_step(
        self,
        obs: List[NDArray],
        action: List[NDArray],
        done: List[bool],
        reward: List[float],
        **kwargs: Dict[str, List[NDArray]]
    ):
        transitions = [
            obs, action, done, reward
        ]
        if len(self.extra_keys) > 0:
            transitions.extend([kwargs[k] for k in self.extra_keys])

        for i, transition in enumerate(
            zip(*transitions)
        ):
            self.agent_trajectories[i].append(transition)
        self.n_step += 1

    def to_agent_trajectory(self, max_size_each_trajectory: int = None) -> List[Dict[str, NDArray]]:
        res = []
        for agent, trajectory in enumerate(self.agent_trajectories):
            values = list(zip(*trajectory))
            episode = dict(zip(self.keys, values))
            for k in self.keys:
                if k == "terminals":
                    dtype = np.bool_
                else:
                    dtype = np.float32
                data = np.array(episode[k], dtype=dtype)
                if max_size_each_trajectory is not None:
                    episode[k] = data[:max_size_each_trajectory]
            res.append(episode)
        return res

        
class TrajectoryDatasetWriter:

    def __init__(self):
        self.trajectories: List[Trajectory] = []

    def add_trajectory(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)
        
    def write_dataset(self, env_meta_info: Dict[str, Any], fname, max_size_each_trajectory=None, compression='gzip'):
        # check env_meta info
        assert "env_id" in env_meta_info
        assert "scenario_id" in env_meta_info
        assert "scenario_configs" in env_meta_info

        data = []
        total_steps = 0
        n_agents = None
        keys = None

        for traj in self.trajectories:
            if n_agents is None:
                keys = traj.keys
                n_agents = len(traj.agents)
            assert n_agents == len(traj.agents), "expected agent number is {} while got {}".format(n_agents, len(traj.agents))
            assert set(keys) == set(traj.keys), "expected keys are: {}, while got: {}".format(keys, traj.keys)
            if max_size_each_trajectory is not None:
                total_steps += min(len(traj), max_size_each_trajectory)
            else:
                total_steps += len(traj)
            data.append(traj.to_agent_trajectory(max_size_each_trajectory=max_size_each_trajectory))

        with open(fname, 'wb') as f:
            pickle.dump({
                'meta_info': {
                    'total_episode': len(self.trajectories),
                    'total_steps': total_steps,
                    'n_agents': n_agents,
                    'keys': tuple(keys),
                    **env_meta_info
                },
                'trajectories': data
            }, f)
