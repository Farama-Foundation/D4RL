import argparse

import numpy as np
from offline_rl.gym_minigrid import fourroom_controller
from offline_rl.gym_minigrid.envs import fourrooms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_episodes", type=int, default=100, help="Num trajs to collect"
    )
    args = parser.parse_args()

    np.random.seed(0)

    env = fourrooms.FourRoomsEnv()
    env.seed(0)
    controller = fourroom_controller.FourRoomController()
    controller.set_target(env.get_target())

    ravg = []
    for _ in range(args.num_episodes):
        env.reset()
        returns = 0
        for t in range(50):
            act, done = controller.get_action(env.agent_pos, env.agent_dir)
            ns, rew, _, _ = env.step(act)
            returns += rew
        ravg.append(returns)
    print("returns", np.mean(ravg))


if __name__ == "__main__":
    main()
