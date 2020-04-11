import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
import locomotion 

from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
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
    # behavior_policy = TanhGaussianPolicy(
    #     obs_dim=obs_dim,
    #     action_dim=action_dim,
    #     hidden_sizes=[M, M],
    # )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        sparse_reward=False,
        target_goal=eval_env.unwrapped.wrapped_env.target_goal,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        sparse_reward=False,
        target_goal=eval_env.unwrapped.wrapped_env.target_goal,
    )
    
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        with_per=False,
    )
    if variant['load_buffer']:
        load_hdf5(eval_env.unwrapped.get_dataset(), replay_buffer)

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        behavior_policy=None,
        **variant['trainer_kwargs']
    )
    print(variant['algorithm_kwargs'])
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    print('training!')
    algorithm.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAC-BEAR')
    parser.add_argument("--exp_type", type=str, default='antmaze-small')
    parser.add_argument("--load_buffer", type=int, default=0)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--max_path_length', default=700, type=int)
    args = parser.parse_args()

    algorithm = 'SAC-online'
    if args.load_buffer:
        algorithm = 'SAC-offline'

    # noinspection PyTypeChecker
    variant = dict(
        algorithm=algorithm,
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        load_buffer=args.load_buffer,
        env_name=args.exp_type,
        algorithm_kwargs=dict(
            num_epochs=2000,
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
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            target_update_method='default',

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

            # What kind of update to use for the policy update
            use_snips_update=False,
            use_behavior_policy_for_base_density=False,

            # Normed Q-values or not to be used
            use_normed_q_values=False,

            # Policy eval
            do_policy_eval=False,
            policy_eval_start=0,
            num_qs=2,

            # PER
            with_per=False,
        ),
    )
    
    setup_logger('sac_maze_final', variant=variant, base_log_dir='./data')
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)