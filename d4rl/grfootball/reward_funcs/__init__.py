from typing import Type
from .basic import Rewarder


def get_reward_func(_type: str) -> Type:
    if _type == "basic":
        return Rewarder
    else:
        raise ValueError("Unregistered reward type: {}".format(_type))