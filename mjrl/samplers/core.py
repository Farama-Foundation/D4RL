import logging
import numpy as np
from mjrl.utils.gym_env import GymEnv
from mjrl.utils import tensor_utils
logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
logging.disable(logging.CRITICAL)


# Single core rollout to sample trajectories
# =======================================================
def do_rollout(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        env_kwargs=None,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :param env_kwargs:  dictionary with parameters, will be passed to env generator
    :return:
    """

    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError

    if base_seed is not None:
        env.set_seed(base_seed)
        np.random.seed(base_seed)
    else:
        np.random.seed()
    horizon = min(horizon, env.horizon)
    paths = []

    for ep in range(num_traj):
        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            env.set_seed(seed)
            np.random.seed(seed)

        observations=[]
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []

        o = env.reset()
        done = False
        t = 0

        while t < horizon and done != True:
            a, agent_info = policy.get_action(o)
            if eval_mode:
                a = agent_info['evaluation']
            env_info_base = env.get_env_infos()
            next_o, r, done, env_info_step = env.step(a)
            # below is important to ensure correct env_infos for the timestep
            env_info = env_info_step if env_info_base == {} else env_info_base
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done
        )
        paths.append(path)

    del(env)
    return paths


def sample_paths(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        max_process_time=300,
        max_timeouts=4,
        suppress_print=False,
        env_kwargs=None,
        ):

    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs)
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)

    # do multiprocessing otherwise
    paths_per_cpu = int(np.ceil(num_traj/num_cpu))
    input_dict_list= []
    for i in range(num_cpu):
        input_dict = dict(num_traj=paths_per_cpu, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon,
                          base_seed=base_seed + i * paths_per_cpu,
                          env_kwargs=env_kwargs)
        input_dict_list.append(input_dict)
    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(do_rollout, input_dict_list,
                                num_cpu, max_process_time, max_timeouts)
    paths = []
    # result is a paths type and results is list of paths
    for result in results:
        for path in result:
            paths.append(path)  

    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " %(timer.time()-start_time) )

    return paths


def sample_data_batch(
        num_samples,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        paths_per_call = 1,
        env_kwargs=None,
        ):

    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    start_time = timer.time()
    print("####### Gathering Samples #######")
    sampled_so_far = 0
    paths_so_far = 0
    paths = []
    base_seed = 123 if base_seed is None else base_seed
    while sampled_so_far < num_samples:
        base_seed = base_seed + 12345
        new_paths = sample_paths(paths_per_call * num_cpu, env, policy,
                                 eval_mode, horizon, base_seed, num_cpu,
                                 suppress_print=True, env_kwargs=env_kwargs)
        for path in new_paths:
            paths.append(path)
        paths_so_far += len(new_paths)
        new_samples = np.sum([len(p['rewards']) for p in new_paths])
        sampled_so_far += new_samples
    print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time() - start_time))
    print("................................. | >>>> # samples = %i # trajectories = %i " % (
    sampled_so_far, paths_so_far))
    return paths


def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):
    
    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts-1)

    pool.close()
    pool.terminate()
    pool.join()  
    return results
