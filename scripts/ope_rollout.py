"""
This script runs rollouts on the OPE policies
using the ONNX runtime and averages the returns.
"""
import argparse

import gym
import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument(
    "policy", type=str, help="ONNX policy file. i.e. cheetah.sampler.onnx"
)
parser.add_argument("env_name", type=str, help="Env name")
parser.add_argument(
    "--num_rollouts", type=int, default=10, help="Number of rollouts to run."
)
args = parser.parse_args()

env = gym.make(args.env_name)

policy = ort.InferenceSession(args.policy)

all_returns = []
for _ in range(args.num_rollouts):
    s = env.reset()
    returns = 0
    for t in range(env._max_episode_steps):
        obs_input = np.expand_dims(s, axis=0).astype(np.float32)
        noise_input = np.random.randn(1, env.action_space.shape[0]).astype(np.float32)
        action, _, _ = policy.run(
            None, {"observations": obs_input, "noise": noise_input}
        )
        s, r, d, _ = env.step(action)
        returns += r
    print(returns, end="\r")
    all_returns.append(returns)
print(args.env_name, ":", np.mean(returns))
