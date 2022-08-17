import gym
from .offline_env import OfflineEnv


class ModularRlEnv(gym.Env):
    def __init__(self,
                 game,
                 **kwargs):
        env = gym.make(game)

        self._env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return self._env.step(action)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def render(self, mode='human'):
        self._env.render(mode)

    def seed(self, seed=None):
        super().seed(seed)
        self._env.seed(seed)


class OfflineModularRlEnv(ModularRlEnv, OfflineEnv):
    def __init__(self, **kwargs):
        game = kwargs['game']
        del kwargs['game']
        ModularRlEnv.__init__(self, game=game)
        OfflineEnv.__init__(self, game=game, **kwargs)