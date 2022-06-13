from typing import Type

from .basic import FeatureEncoder


def get_encoder(_type: str) -> Type:
    if _type == "basic":
        return FeatureEncoder
    else:
        raise ValueError("Unresigitered encoder type: {}".format(_type))