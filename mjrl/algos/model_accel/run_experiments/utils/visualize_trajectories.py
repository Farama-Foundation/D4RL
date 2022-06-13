import pickle
import click
import json
import numpy as np
import torch
import mjrl.envs
import trajopt.envs
import mj_envs
import mjrl.utils.tensor_utils as tensor_utils

from mjrl.utils.gym_env import GymEnv
from mjrl.algos.model_accel.sampling import evaluate_policy

DESC = '''
Helper script to visualize optimized trajectories (list of trajectories in trajopt format).\n
USAGE:\n
    $ python viz_trajectories.py --file path_to_file.pickle\n
'''
@click.command(help=DESC)
@click.option('--file', type=str, help='pickle file with trajectories', required= True)
@click.option('--seed', type=int, default=123)
@click.option('--noise_level', type=float, default=0.0)
@click.option('--num_episodes', type=int, help='number of times to play trajectories', default=5)
@click.option('--config', type=str, help='if provided MPC params from here will be used.', default=None)
@click.option('--device_path', type=str, default=None)
def main(file, seed, noise_level, num_episodes, config, device_path):
    exp_data = pickle.load(open(file, 'rb'))
    policy = exp_data['policy']
    model = exp_data['fitted_model']
    model = model[-1] if type(model) == list else model
    env_id = policy.env.env_id
    render = True

    # TODO(Aravind): Map to hardware if device_path is specified

    env = GymEnv(env_id)
    policy.env = env

    env.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if config is not None:
        try:
            with open(config, 'r') as f:
                config = eval(f.read())
        except:
            with open(config, 'r') as f:
                config = json.load(f)
        policy.plan_horizon = config['plan_horizon']
        policy.num_traj = config['plan_paths']
        policy.kappa = config['kappa']
        policy.filter_coefs = [config['filter_coefs'][k] for k in ['f1', 'f2', 'f3', 'f4']]
        policy.omega = config['omega'] if 'omega' in config.keys() else 0.0

    # TODO(Aravind): Implement capability to set predicted state for rendering purposes
    # evaluate_policy(env, policy, model, noise_level, real_step=False, num_episodes=num_episodes, visualize=render)
    evaluate_policy(env, policy, model, noise_level, real_step=True, num_episodes=num_episodes, visualize=render)

    # final close out
    env.reset()


if __name__ == '__main__':
    main()
