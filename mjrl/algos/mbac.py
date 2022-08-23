import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
from tqdm import tqdm
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.mpc_actor import MPCActor
from mjrl.algos.behavior_cloning import BC


class MBAC(BC):
    def __init__(self,
                 env_name,
                 policy,
                 expert_paths = None, # for the initial seeding
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 optimizer = None,
                 loss_type = 'MSE',  # can be 'MLE' or 'MSE'
                 seed = 123,
                 buffer_size = 50,   # measured in number of trajectories
                 mpc_params = None,
                 save_logs = True,
                 ):

        super().__init__(expert_paths=expert_paths,
                         policy=policy,
                         epochs=epochs,
                         batch_size=batch_size,
                         lr=lr,
                         optimizer=optimizer,
                         loss_type=loss_type,
                         save_logs=save_logs,
                         )
        self.expert_paths = [] if self.expert_paths is None else self.expert_paths
        self.buffer_size = buffer_size

        # For the MPC policy
        self.env = GymEnv(env_name)
        self.env.reset(seed=seed)
        if mpc_params is None:
            mean = np.zeros(self.env.action_dim)
            sigma = 1.0 * np.ones(self.env.action_dim)
            filter_coefs = [sigma, 0.05, 0.0, 0.0]
            mpc_params = dict(env=GymEnv(env_name), H=10,
                              paths_per_cpu=25, num_cpu=1,
                              kappa=10.0, gamma=1.0,
                              mean=mean, filter_coefs=filter_coefs,
                              seed=seed)
        else:
            mpc_params['env'] = GymEnv(env_name)
            mpc_params['seed'] = seed

        self.mpc_params = mpc_params
        self.mpc_policy = MPCActor(**mpc_params)

    def collect_paths(self, num_traj=10,
                      mode='policy',
                      horizon=None,
                      render=False
                      ):
        horizon = self.env.horizon if horizon is None else horizon
        paths = []
        for i in tqdm(range(num_traj)):
            self.env.reset()
            obs, act_pi, act_mpc, rew, states = [], [], [], [], []
            for t in range(horizon):
                o = self.env.get_obs()
                s = self.env.get_env_state()
                a_pi = self.policy.get_action(o)[0]
                a_mpc = self.mpc_policy.get_action(s)
                a = a_pi if mode == 'policy' else a_mpc
                next_o, r, done, _ = self.env.step(a)
                if render:
                    self.env.render()
                # store data
                obs.append(o)
                rew.append(r)
                states.append(s)
                act_pi.append(a_pi)
                act_mpc.append(a_mpc)
                # kill if done
                if done:
                    break
            path = dict(observations=np.array(obs),
                        actions=np.array(act_pi),
                        expert_actions=np.array(act_mpc),
                        rewards=np.array(rew),
                        states=states,
                        )
            paths.append(path)
        return paths

    def add_paths_to_buffer(self, paths):
        for path in paths:
            self.expert_paths.append(path)
        if len(self.expert_paths) > self.buffer_size:
            # keep recent trajectories
            # TODO: Also consider keeping best performing trajectories
            self.expert_paths = self.expert_paths[-self.buffer_size:]
        if self.save_logs:
            self.logger.log_kv('buffer_size', len(self.expert_paths))

    def get_data_from_buffer(self):
        observations = np.concatenate([path["observations"] for path in self.expert_paths])
        expert_actions = np.concatenate([path["expert_actions"] for path in self.expert_paths])
        observations = torch.Tensor(observations).float()
        expert_actions = torch.Tensor(expert_actions).float()
        data = dict(observations=observations, expert_actions=expert_actions)
        return data

    def train_step(self, num_traj=10, **kwargs):
        # collect data using policy actions
        # fit policy to expert actions on these states
        new_paths = self.collect_paths(num_traj, mode='policy')
        self.add_paths_to_buffer(new_paths)
        data = self.get_data_from_buffer()
        self.fit(data, **kwargs)
        stoc_pol_perf = np.mean([np.sum(path['rewards']) for path in new_paths])
        return stoc_pol_perf