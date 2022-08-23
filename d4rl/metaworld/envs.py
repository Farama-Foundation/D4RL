import gym
import random
from .offline_env import OfflineEnv
import metaworld


class MWRlEnv(gym.Env):
    def __init__(self,
                 game,
                 **kwargs):

        mt1 = metaworld.MT1(game)
        self._env = env = mt1.train_classes[game]() 
        self.tasks = mt1.train_tasks

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        # action should be an int here
        obs, reward, _, info = self._env.step(action)

        done = self.curr_episode_steps >= 500

        self.curr_episode_steps += 1

        return obs, reward, done, info

    def reset(self, **kwargs):
        task = random.choice(self.tasks)
        self._env.set_task(task)
        obs = self._env.reset()
        self.curr_episode_steps = 0
        return obs

    def render(self, mode='human'):
        self._env.render(mode)

    def seed(self, seed=None):
        super().seed(seed)
        self._env.seed(seed)


class OfflineMWRlEnv(MWRlEnv, OfflineEnv):
    def __init__(self, **kwargs):
        game = kwargs['game']
        del kwargs['game']
        MWRlEnv.__init__(self, game=game)
        OfflineEnv.__init__(self, game=game, **kwargs)