"""
convenience function to generate env
useful if we want some procedural env generation
"""

import gym
from mjrl.utils.gym_env import GymEnv

def get_environment(env_name=None, **kwargs):
    if env_name is None: print("Need to specify environment name")
    e = GymEnv(env_name)
    # can make procedural modifications here if needed using kwargs
    return e
