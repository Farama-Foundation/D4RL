from argparse import Namespace

from .env import GRFootball
from d4rl import offline_env


class OfflineGRFEnv(GRFootball, offline_env.OfflineEnv):

    def __init__(self, **kwargs):
        scenario_config = kwargs["scenario_config"]
        args = Namespace(**scenario_config)
        GRFootball.__init__(self, args)
        offline_env.OfflineEnv.__init__(
            self,
            dataset_url=kwargs.get("dataset_url", None),
            ref_max_score=kwargs.get("ref_max_score", None),
            ref_min_score=kwargs.get("ref_min_score", None),
            deprecated=kwargs.get("deprecated", False),
            deprecation_message=kwargs.get("deprecation_message", None)
        )


def get_env(**kwargs):
    return OfflineGRFEnv(**kwargs)