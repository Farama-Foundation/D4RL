import gym
from .offline_env import OfflineEnv
from gym import spaces
import numpy as np
import copy

class GatoAtariObsWrapper(gym.ObservationWrapper):
    """Wrap observation of Atari games for Gato pretraining
    1. change 84*84 image to 80 * 80 so as to make it divided by 16
    2. merge grayscale image into 3-dimensional RGB channels
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = copy.deepcopy(env.observation_space)
        img_sp = self.observation_space["image"]
        self.observation_space["image"] = spaces.Box(
            0, 256, shape=(np.prod(img_sp.shape),), dtype=np.float32)
    def observation(self, obs):
        obs["image"] = obs["image"].flatten().astype(np.float32)
        return obs

def post_process(obs):
    bsz, *_ = obs["image"].shape
    obs["image"] = obs["image"].reshape(bsz, -1).astype(np.float32)
    return obs

class BabyaiEnv(gym.Env):
    def __init__(self,
                 game,
                 **kwargs):
        env = gym.make(game)
        env = GatoAtariObsWrapper(env)
        self._env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.post_process_fn = post_process

    def step(self, action):
        return self._env.step(action)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def render(self, mode='human'):
        self._env.render(mode)

    def seed(self, seed=None):
        super().seed(seed)
        self._env.seed(seed)


class OfflineBabyaiEnv(BabyaiEnv, OfflineEnv):
    def __init__(self, **kwargs):
        game = kwargs['game']
        del kwargs['game']
        BabyaiEnv.__init__(self, game=game)
        OfflineEnv.__init__(self, game=game, **kwargs)
