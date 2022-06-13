"""
Job script to optimize trajectories with fitted model
"""

import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
import mjrl.envs
import time as timer
import argparse
import os
import json
import mjrl.samplers.core as trajectory_sampler
import mjrl.utils.tensor_utils as tensor_utils
from tqdm import tqdm
from tabulate import tabulate
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.logger import DataLog
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.algos.model_accel.nn_dynamics import DynamicsModel
from mjrl.algos.model_accel.model_learning_mpc import MPCPolicy
from mjrl.algos.model_accel.sampling import sample_paths, evaluate_policy


# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Trajectory Optimization with fitted models.')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
args = parser.parse_args()
OUT_DIR = args.output
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
with open(args.config, 'r') as f:
    job_data = eval(f.read())

# Unpack args and make files for easy access
logger = DataLog()
ENV_NAME = job_data['env_name']
PICKLE_FILE = OUT_DIR + '/exp_results.pickle'
EXP_FILE = OUT_DIR + '/job_data.json'
SEED = job_data['seed']
job_data['filter_coefs'] = [job_data['filter_coefs'][k] for k in ['f1', 'f2', 'f3', 'f4']]

# base cases
if 'num_models' not in job_data.keys():
    job_data['num_models'] = 1
if job_data['num_models'] == 1 or 'omega' not in job_data.keys():
    job_data['omega'] = 0.0
if 'eval_rollouts' not in job_data.keys():
    job_data['eval_rollouts'] = 0
if 'save_freq' not in job_data.keys():
    job_data['save_freq'] = 10
if 'device' not in job_data.keys():
    job_data['device'] = 'cpu'
if 'debug_mode' in job_data.keys():
    DEBUG = job_data['debug_mode']
else:
    DEBUG =False
if 'device_path' not in job_data.keys():
    job_data['device_path'] = None
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)

del(job_data['seed'])
job_data['base_seed'] = SEED

# ===============================================================================
# Train loop
# ===============================================================================

np.random.seed(SEED)
torch.random.manual_seed(SEED)

# TODO(Aravind): Map to hardware if device_path is specified

e = GymEnv(ENV_NAME)
e.set_seed(SEED)
models = [DynamicsModel(state_dim=e.observation_dim, act_dim=e.action_dim, seed=SEED+i, **job_data)
          for i in range(job_data['num_models'])]
exploratory_policy = MLP(e.spec, seed=SEED, init_log_std=job_data['init_log_std'])
paths = []

for outer_iter in range(job_data['num_iter']):

    ts = timer.time()
    print("================> ITERATION : %i " % outer_iter)
    print("Getting interaction data from real dynamics ...")

    if outer_iter == 0:
        iter_paths = trajectory_sampler.sample_paths(job_data['n_init_paths'], e,
                                                     exploratory_policy,
                                                     eval_mode=False, base_seed=SEED)
    else:
        iter_paths = sample_paths(job_data['paths_per_iter'],
                                  mpc_policy.env, mpc_policy,
                                  eval_mode=(not job_data['noisy_mpc']),
                                  noise_level=job_data['noise_level'],
                                  base_seed=SEED + outer_iter)

    # reset the environment (good for hardware)
    e.reset()

    for p in iter_paths:
        paths.append(p)

    if len(paths) > job_data['max_paths']:
        diff = len(paths) - job_data['max_paths']
        paths[:diff] = []

    s = np.concatenate([p['observations'][:-1] for p in paths])
    a = np.concatenate([p['actions'][:-1] for p in paths])
    sp = np.concatenate([p['observations'][1:] for p in paths])
    r = np.array([np.sum(p['rewards']) for p in iter_paths])
    rollout_score = np.mean(r)

    logger.log_kv('fit_epochs', job_data['fit_epochs'])
    logger.log_kv('rollout_score', rollout_score)
    try:
        rollout_metric = e.env.env.evaluate_success(iter_paths)
        logger.log_kv('rollout_metric', rollout_metric)
    except:
        pass

    print("Data gathered, fitting model ...")
    if job_data['refresh_fit']:
        models = [DynamicsModel(state_dim=e.observation_dim, act_dim=e.action_dim, seed=SEED+123*outer_iter,
                                **job_data) for i in range(job_data['num_models'])]

    for i, model in enumerate(models):
        epoch_loss = model.fit(s, a, sp, job_data['fit_mb_size'], job_data['fit_epochs'])
        logger.log_kv('loss_before_' + str(i), epoch_loss[0])
        logger.log_kv('loss_after_' + str(i), epoch_loss[-1])

    mpc_policy = MPCPolicy(env=e, fitted_model=models, seed=SEED+12345*outer_iter, **job_data)

    if job_data['eval_rollouts'] > 0:
        print("Performing validation rollouts ... ")
        eval_paths = evaluate_policy(mpc_policy.env, mpc_policy, mpc_policy.fitted_model[0], noise_level=0.0,
                                     real_step=True, num_episodes=job_data['eval_rollouts'], visualize=False)
        eval_score = np.mean([np.sum(p['rewards']) for p in eval_paths])
        logger.log_kv('eval_score', eval_score)
        try:
            eval_metric = e.env.env.evaluate_success(eval_paths)
            logger.log_kv('eval_metric', eval_metric)
        except:
            pass
    else:
        eval_paths = []

    exp_data = dict(policy=mpc_policy, fitted_model=mpc_policy.fitted_model,
                    log=logger.log, rollout_paths=iter_paths, eval_paths=eval_paths)
    if outer_iter > 0 and outer_iter % job_data['save_freq'] == 0:
        pickle.dump(exp_data, open(PICKLE_FILE, 'wb'))
        pickle.dump(exp_data, open(OUT_DIR + '/iteration_' + str(outer_iter) + '.pickle', 'wb'))

    tf = timer.time()
    logger.log_kv('iter_time', tf-ts)
    print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                               logger.get_current_log().items()))
    print(tabulate(print_data))
    logger.save_log(OUT_DIR+'/')
    make_train_plots(log=logger.log, keys=['rollout_score', 'eval_score', 'rollout_metric', 'eval_metric'],
                     save_loc=OUT_DIR+'/')

    if job_data['debug_mode']:
        evaluate_policy(e, mpc_policy, mpc_policy.fitted_model[0], job_data['noise_level'], False, 5, True)
        evaluate_policy(e, mpc_policy, mpc_policy.fitted_model[0], job_data['noise_level'], True, 5, True)

    pickle.dump(exp_data, open(PICKLE_FILE, 'wb')) # final save