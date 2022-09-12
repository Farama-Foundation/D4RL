import click
from mjrl.utils.gym_env import GymEnv

# from mjrl.policies.gaussian_mlp import MLP

DESC = """
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python visualize_policy.py --env_name door-v0 \n
    $ python visualize_policy.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
"""


class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        return [
            self.env.action_space.sample(),
            {"evaluation": self.env.action_space.sample()},
        ]


# MAIN =========================================================
@click.command(help=DESC)
@click.option("--env_name", type=str, help="environment to load", required=True)
@click.option(
    "--policy", type=str, help="absolute path of the policy file", default=None
)
@click.option(
    "--mode",
    type=str,
    help="exploration or evaluation mode for policy",
    default="evaluation",
)
@click.option(
    "--seed", type=int, help="seed for generating environment instances", default=123
)
@click.option(
    "--episodes", type=int, help="number of episodes to visualize", default=10
)
def main(env_name, policy, mode, seed, episodes):
    e = GymEnv(env_name)
    e.set_seed(seed)
    """
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=-1.0)
    """
    pi = RandomPolicy(e)
    # render policy
    e.visualize_policy(pi, num_episodes=episodes, horizon=e.horizon, mode=mode)


if __name__ == "__main__":
    main()
