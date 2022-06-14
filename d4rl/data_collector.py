from typing import Any, List, Dict

import gym


class Collector:

    def __init__(self, env: gym.Env = None, policy: Any = None):
        if policy is not None:
            assert hasattr("compute_action", policy)
        if env is not None:
            assert isinstance(env, gym.Env), type(env)

        self.env = env
        self.policy = policy

    def pass_batch_to_save(self, batches: List[Any], file_path: str, parser: Any):
        raise NotImplementedError

    def run_and_save(self, keys_to_track: List[str], num_epiosde: int, file_path: str) -> Dict[str, Any]:
        """Run some episodes to collect data.

        Args:
            keys_to_track (List[str]): A list of keys to trace.
            num_epiosde (int): The number of episodes.
            file_path (str): File path for data saving. Should be hdf.

        Returns:
            Dict[str, Any]: ...
        """
        raise NotImplementedError
