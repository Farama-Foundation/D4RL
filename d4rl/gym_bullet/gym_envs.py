from .. import offline_env
from pybullet_envs.gym_locomotion_envs import HopperBulletEnv, HalfCheetahBulletEnv, Walker2DBulletEnv, AntBulletEnv
from ..utils.wrappers import NormalizedBoxEnv

class OfflineAntEnv(AntBulletEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        AntBulletEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHopperEnv(HopperBulletEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HopperBulletEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHalfCheetahEnv(HalfCheetahBulletEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HalfCheetahBulletEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineWalker2dEnv(Walker2DBulletEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        Walker2DBulletEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)


def get_ant_env(**kwargs):
    return NormalizedBoxEnv(OfflineAntEnv(**kwargs))

def get_halfcheetah_env(**kwargs):
    return NormalizedBoxEnv(OfflineHalfCheetahEnv(**kwargs))

def get_hopper_env(**kwargs):
    return NormalizedBoxEnv(OfflineHopperEnv(**kwargs))

def get_walker2d_env(**kwargs):
    return NormalizedBoxEnv(OfflineWalker2dEnv(**kwargs))

