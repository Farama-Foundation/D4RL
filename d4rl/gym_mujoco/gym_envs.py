import numpy as np
import gym
from gym.envs.mujoco import HalfCheetahEnv, AntEnv, HopperEnv, Walker2dEnv

from .. import offline_env
from ..utils.wrappers import NormalizedBoxEnv

class OfflineAntEnv(AntEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        AntEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHopperEnv(HopperEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HopperEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHalfCheetahEnv(HalfCheetahEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HalfCheetahEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineWalker2dEnv(Walker2dEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        Walker2dEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)


class MJCVisionWrapper(gym.ObservationWrapper):
    def __init__(self, env, image_size=84):
        super(MJCVisionWrapper, self).__init__(env)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(image_size, image_size, 3),
                dtype=np.float32)

    def observation(self, s):
        image = self.env.render('rgb_array', width=self.image_size, height=self.image_size)
        image = image / 255.0
        return image


def get_ant_env(**kwargs):
    return NormalizedBoxEnv(OfflineAntEnv(**kwargs))

def get_cheetah_env(**kwargs):
    return NormalizedBoxEnv(OfflineHalfCheetahEnv(**kwargs))

def get_hopper_env(**kwargs):
    return NormalizedBoxEnv(OfflineHopperEnv(**kwargs))

def get_walker_env(**kwargs):
    return NormalizedBoxEnv(OfflineWalker2dEnv(**kwargs))

def get_ant_vision_env(**kwargs):
    return MJCVisionWrapper(NormalizedBoxEnv(OfflineAntEnv(**kwargs)))

def get_halfcheetah_vision_env(**kwargs):
    return MJCVisionWrapper(NormalizedBoxEnv(OfflineHalfCheetahEnv(**kwargs)))

def get_hopper_vision_env(**kwargs):
    return MJCVisionWrapper(NormalizedBoxEnv(OfflineHopperEnv(**kwargs)))

def get_walker2d_vision_env(**kwargs):
    return MJCVisionWrapper(NormalizedBoxEnv(OfflineWalker2dEnv(**kwargs)))

if __name__ == '__main__':
    """Example usage of these envs"""
    pass
