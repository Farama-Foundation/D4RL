import logging
logging.disable(logging.CRITICAL)

from tabulate import tabulate
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
import numpy as np
import pickle
import time as timer
import os
import copy


def _load_latest_policy_and_logs(agent, *, policy_dir, logs_dir):
    """Loads the latest policy.
    Returns the next step number to begin with.
    """
    assert os.path.isdir(policy_dir), str(policy_dir)
    assert os.path.isdir(logs_dir), str(logs_dir)

    log_csv_path = os.path.join(logs_dir, 'log.csv')
    if not os.path.exists(log_csv_path):
        return 0   # fresh start

    print("Reading: {}".format(log_csv_path))
    agent.logger.read_log(log_csv_path)
    last_step = agent.logger.max_len - 1
    if last_step <= 0:
        return 0   # fresh start


    # find latest policy/baseline
    i = last_step
    while i >= 0:
        policy_path = os.path.join(policy_dir, 'policy_{}.pickle'.format(i))
        baseline_path = os.path.join(policy_dir, 'baseline_{}.pickle'.format(i))

        if not os.path.isfile(policy_path):
            i = i -1
            continue
        else:
            print("Loaded last saved iteration: {}".format(i))

        with open(policy_path, 'rb') as fp:
            agent.policy = pickle.load(fp)
        with open(baseline_path, 'rb') as fp:
            agent.baseline = pickle.load(fp)

        # additional
        # global_status_path = os.path.join(policy_dir, 'global_status.pickle')
        # with open(global_status_path, 'rb') as fp:
        #     agent.load_global_status( pickle.load(fp) )

        agent.logger.shrink_to(i + 1)
        assert agent.logger.max_len == i + 1
        return agent.logger.max_len

    # cannot find any saved policy
    raise RuntimeError("Log file exists, but cannot find any saved policy.")

def train_agent(job_name, agent,
                seed = 0,
                niter = 101,
                gamma = 0.995,
                gae_lambda = None,
                num_cpu = 1,
                sample_mode = 'trajectories',
                num_traj = 50,
                num_samples = 50000, # has precedence, used with sample_mode = 'samples'
                save_freq = 10,
                evaluation_rollouts = None,
                plot_keys = ['stoc_pol_mean'],
                ):

    np.random.seed(seed)
    if os.path.isdir(job_name) == False:
        os.mkdir(job_name)
    previous_dir = os.getcwd()
    os.chdir(job_name) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False and agent.save_logs == True: os.mkdir('logs')
    best_policy = copy.deepcopy(agent.policy)
    best_perf = -1e8
    train_curve = best_perf*np.ones(niter)
    mean_pol_perf = 0.0
    e = GymEnv(agent.env.env_id)

    # Load from any existing checkpoint, policy, statistics, etc.
    # Why no checkpointing.. :(
    i_start = _load_latest_policy_and_logs(agent,
                                           policy_dir='iterations',
                                           logs_dir='logs')
    if i_start:
        print("Resuming from an existing job folder ...")

    for i in range(i_start, niter):
        print("......................................................................................")
        print("ITERATION : %i " % i)

        if train_curve[i-1] > best_perf:
            best_policy = copy.deepcopy(agent.policy)
            best_perf = train_curve[i-1]

        N = num_traj if sample_mode == 'trajectories' else num_samples
        args = dict(N=N, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu)
        stats = agent.train_step(**args)
        train_curve[i] = stats[0]

        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths(num_traj=evaluation_rollouts, policy=agent.policy, num_cpu=num_cpu,
                                      env=e.env_id, eval_mode=True, base_seed=seed)
            mean_pol_perf = np.mean([np.sum(path['rewards']) for path in eval_paths])
            if agent.save_logs:
                agent.logger.log_kv('eval_score', mean_pol_perf)
                try:
                    eval_success = e.env.env.evaluate_success(eval_paths)
                    agent.logger.log_kv('eval_success', eval_success)
                except:
                    pass

        if i % save_freq == 0 and i > 0:
            if agent.save_logs:
                agent.logger.save_log('logs/')
                make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
            policy_file = 'policy_%i.pickle' % i
            baseline_file = 'baseline_%i.pickle' % i
            pickle.dump(agent.policy, open('iterations/' + policy_file, 'wb'))
            pickle.dump(agent.baseline, open('iterations/' + baseline_file, 'wb'))
            pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
            # pickle.dump(agent.global_status, open('iterations/global_status.pickle', 'wb'))

        # print results to console
        if i == 0:
            result_file = open('results.txt', 'w')
            print("Iter | Stoc Pol | Mean Pol | Best (Stoc) \n")
            result_file.write("Iter | Sampling Pol | Evaluation Pol | Best (Sampled) \n")
            result_file.close()
        print("[ %s ] %4i %5.2f %5.2f %5.2f " % (timer.asctime(timer.localtime(timer.time())),
                                                 i, train_curve[i], mean_pol_perf, best_perf))
        result_file = open('results.txt', 'a')
        result_file.write("%4i %5.2f %5.2f %5.2f \n" % (i, train_curve[i], mean_pol_perf, best_perf))
        result_file.close()
        if agent.save_logs:
            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                       agent.logger.get_current_log().items()))
            print(tabulate(print_data))

    # final save
    pickle.dump(best_policy, open('iterations/best_policy.pickle', 'wb'))
    if agent.save_logs:
        agent.logger.save_log('logs/')
        make_train_plots(log=agent.logger.log, keys=plot_keys, save_loc='logs/')
    os.chdir(previous_dir)
