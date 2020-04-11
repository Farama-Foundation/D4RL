import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
import locomotion 

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, VAEPolicy
from rlkit.torch.sac.bear import BEARTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import h5py
import numpy as np
import gym

import argparse, os

def load_hdf5(dataset, replay_buffer):
    all_obs = dataset['observations']
    all_act = dataset['actions']
    N = min(all_obs.shape[0], 1000000)
    _obs = all_obs[:N]
    _actions = all_act[:N]
    _next_obs = np.concatenate([all_obs[1:N,:], np.zeros_like(_obs[0])[np.newaxis,:]], axis=0)
    _rew = dataset['rewards'][:N]
    _done = dataset['terminals'][:N]

    replay_buffer._observations = _obs
    replay_buffer._next_obs = _next_obs
    replay_buffer._actions = _actions
    replay_buffer._rewards = np.expand_dims(_rew, 1)
    replay_buffer._terminals = np.expand_dims(_done, 1)
    replay_buffer._size = N-1
    replay_buffer._top = replay_buffer._size

def experiment(variant):
    expl_env = gym.make(variant['env_name'])
    eval_env = expl_env

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M,],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M,],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M,],    # Making it easier to visualize
    )
    vae_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        latent_dim=action_dim * 2,
    )
    eval_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    load_hdf5(eval_env.unwrapped.get_dataset(), replay_buffer)
    
    trainer = BEARTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae_policy,
        **variant['trainer_kwargs']
    )
    # variant['algorithm_kwargs']['max_path_length'] = expl_env._max_episode_steps
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        batch_rl=True,
        q_learning_alg=True,            ### SET THIS TO TRUE, BEAR is a Q-learning algorithm
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC-BEAR')
    parser.add_argument('--exp_type', default='antmaze-small', help='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--qf_lr', default=3e-4, type=float)
    parser.add_argument('--policy_lr', default=3e-4, type=float)
    parser.add_argument('--mmd_sigma', default=20, type=float)
    parser.add_argument('--kernel_type', default='gaussian', type=str)
    parser.add_argument('--max_path_length', default=700, type=int)
    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="BEAR",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        env_name=args.exp_type,
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=args.max_path_length,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,
            target_update_method='default',

            # BEAR specific params
            mode='auto',
            kernel_choice=args.kernel_type,
            policy_update_style='0',
            mmd_sigma=args.mmd_sigma,
            target_mmd_thresh=0.05,

            # gradient penalty hparams
            with_grad_penalty_v1=False,
            with_grad_penalty_v2=False,
            grad_coefficient_policy=0.001,
            grad_coefficient_q=1E-4,
            start_epoch_grad_penalty=24000,

            # Target nets/ policy vs Q-function update
            use_target_nets=True,
            policy_update_delay=1,
            num_steps_policy_update_only=1,
            do_policy_eval=False,
        ),
    )
    setup_logger('maze_BEAR_port_code_ant', variant=variant, base_log_dir='./data')
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)