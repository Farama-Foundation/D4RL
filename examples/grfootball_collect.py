"""Data collection examples. Google research football.

@author: Ming Zhou
@date: 2022/06/15 08:33 PM
"""

from typing import List
from argparse import ArgumentParser, Namespace

import os
import time
import numpy as np
import gym

from d4rl.grfootball import scenario_configs
from d4rl.utils.dataset_utils import TrajectoryDatasetWriter, Trajectory


D4RL_DATASET_DIR = os.path.expanduser('~/.d4rl/datasets')


def compute_action(
    observation: List[np.ndarray], available_actions: List[np.ndarray]
):
    assert len(observation) == n_agents
    act = []
    for ava_actions in available_actions:
        idxes = np.where(ava_actions == 1)[0]
        act.append(np.random.choice(idxes))
    return act


if __name__ == "__main__":
    parser = ArgumentParser("Data collection procedure for MAPPO on GRF.")

    parser.add_argument("--scenario_id", default="5_vs_5", type=str, help="scenario name.")
    parser.add_argument(
        "--n_episode", default=300, type=int, help="the number of episodes to run."
    )
    parser.add_argument(
        "--max_episode_len", default=3000, type=int, help="max episode length."
    )
    parser.add_argument(
        "--use_builtin_gk", action="store_true", help="use builtin goal keeper or not."
    )
    parser.add_argument(
        "--segment_length", default=100, help="segment length of each dataset."
    )

    args = parser.parse_args()

    env_runtime_configs = scenario_configs[args.scenario_id]
    env_runtime_configs["use_builtin_gk"] = args.use_builtin_gk

    env = gym.make('GRFootball-v0', scenario_config=env_runtime_configs)

    # the numberf controled agents
    n_agents = env.n_agents

    num_actions = 19

    start = time.time()
    n_frame = 0
    writer = TrajectoryDatasetWriter()
    seg_th = 0
    env_meta_info = {
        'env_id': 'GRFootball',
        'scenario_id': args.scenario_id,
        'scenario_configs': scenario_configs[args.scenario_id]
    }
    dataset_dir = os.path.join(D4RL_DATASET_DIR, "gfootball")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for episode_th in range(args.n_episode):
        cnt = 0
        trajectory = Trajectory(
            episode_id=str(time.time()),
            agents=list(range(n_agents)),
            max_episode_length=args.max_episode_len,
            extra_keys=['available_actions']
        )

        observations, states, available_actions = env.reset()
        done = False

        while not done:
            cnt += 1
            n_frame += 1
            actions = compute_action(
                observation=observations,
                available_actions=available_actions
            )

            for action, mask in zip(actions, available_actions):
                assert mask[action] == 1, (action, mask)
            (
                next_observations,
                next_states,
                rewards,
                dones,
                infos,
                next_available_actions,
            ) = env.step(actions)
            done = any(dones) or cnt >= args.max_episode_len

            trajectory.record_step(
                obs=observations,
                action=actions,
                done=[done] * n_agents,
                reward=rewards,
                # extra keys
                # states=states,
                available_actions=available_actions,
            )

            observations = next_observations
            states = next_states
            available_actions = next_available_actions

            if n_frame % 100 == 0:
                cur_time = time.time()
                print(
                    "FPS: {:.3} reward: {:.3f} {:.3f} {:.3f} done: {}".format(
                        n_frame / (cur_time - start),
                        np.mean(rewards),
                        np.max(rewards),
                        np.min(rewards),
                        any(dones),
                    )
                )
        writer.add_trajectory(trajectory)

        if (episode_th + 1) % args.segment_length == 0:
            fname = os.path.join(dataset_dir, f"seg_{seg_th}.pkl")

            print("write dataset to: {} with meta info:\n{}".format(fname, env_meta_info))
            writer.write_dataset(
                env_meta_info=env_meta_info,
                fname=fname,
                flush=True
            )
            seg_th += 1

    if not writer.empty():
        fname = os.path.join(dataset_dir, f"seg_{seg_th}.pkl")
        print("write dataset to: {} with meta info:\n{}".format(fname, env_meta_info))
        writer.write_dataset(
            env_meta_info=env_meta_info,
            fname=fname,
            flush=True
        )
        
        
