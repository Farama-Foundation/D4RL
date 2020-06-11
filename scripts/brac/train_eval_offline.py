# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation in the offline mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

from absl import logging


import gin
import gym
import numpy as np
import tensorflow as tf0
import tensorflow.compat.v1 as tf

from behavior_regularized_offline_rl.brac import dataset
from behavior_regularized_offline_rl.brac import train_eval_utils
from behavior_regularized_offline_rl.brac import utils

from gym.wrappers import time_limit

from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper


def get_offline_data(tf_env):
  gym_env = tf_env.pyenv.envs[0]
  #offline_dataset = gym_env.unwrapped.get_dataset()
  offline_dataset = gym_env.get_dataset()
  dataset_size = len(offline_dataset['observations'])
  tf_dataset = dataset.Dataset(
      tf_env.observation_spec(),
      tf_env.action_spec(),
      size=dataset_size)
  observation_dtype = tf_env.observation_spec().dtype
  action_dtype = tf_env.action_spec().dtype

  offline_dataset['terminals'] = np.squeeze(offline_dataset['terminals'])
  offline_dataset['rewards'] = np.squeeze(offline_dataset['rewards'])
  nonterminal_steps, = np.where(
      np.logical_and(
          np.logical_not(offline_dataset['terminals']),
          np.arange(dataset_size) < dataset_size - 1))
  logging.info('Found %d non-terminal steps out of a total of %d steps.' % (
      len(nonterminal_steps), dataset_size))

  s1 = tf.convert_to_tensor(offline_dataset['observations'][nonterminal_steps],
                            dtype=observation_dtype)
  s2 = tf.convert_to_tensor(offline_dataset['observations'][nonterminal_steps + 1],
                            dtype=observation_dtype)
  a1 = tf.convert_to_tensor(offline_dataset['actions'][nonterminal_steps],
                            dtype=action_dtype)
  a2 = tf.convert_to_tensor(offline_dataset['actions'][nonterminal_steps + 1],
                            dtype=action_dtype)
  discount = tf.convert_to_tensor(
      1. - offline_dataset['terminals'][nonterminal_steps + 1],
      dtype=tf.float32)
  reward = tf.convert_to_tensor(offline_dataset['rewards'][nonterminal_steps],
                                dtype=tf.float32)

  transitions = dataset.Transition(
      s1, s2, a1, a2, discount, reward)

  tf_dataset.add_transitions(transitions)
  return tf_dataset


def env_factory(env_name):
  gym_env = gym.make(env_name)
  gym_spec = gym.spec(env_name)
  if gym_spec.max_episode_steps in [0, None]:  # Add TimeLimit wrapper.
    gym_env = time_limit.TimeLimit(gym_env, max_episode_steps=1000)

  tf_env = tf_py_environment.TFPyEnvironment(
      gym_wrapper.GymWrapper(gym_env))
  return tf_env


@gin.configurable
def train_eval_offline(
    # Basic args.
    log_dir,
    data_file,
    agent_module,
    env_name='HalfCheetah-v2',
    n_train=int(1e6),
    shuffle_steps=0,
    seed=0,
    use_seed_for_data=False,
    # Train and eval args.
    total_train_steps=int(1e6),
    summary_freq=100,
    print_freq=1000,
    save_freq=int(2e4),
    eval_freq=5000,
    n_eval_episodes=20,
    # Agent args.
    model_params=(((200, 200),), 2),
    behavior_ckpt_file=None,
    value_penalty=True,
    #model_params=((200, 200),),
    optimizers=(('adam', 0.001),),
    batch_size=256,
    weight_decays=(0.0,),
    update_freq=1,
    update_rate=0.005,
    discount=0.99,
    ):
  """Training a policy with a fixed dataset."""
  # Create tf_env to get specs.
  print('[train_eval_offline.py] env_name=', env_name)
  print('[train_eval_offline.py] data_file=', data_file)
  print('[train_eval_offline.py] agent_module=', agent_module)
  print('[train_eval_offline.py] model_params=', model_params)
  print('[train_eval_offline.py] optimizers=', optimizers)
  print('[train_eval_offline.py] bckpt_file=', behavior_ckpt_file)
  print('[train_eval_offline.py] value_penalty=', value_penalty)

  tf_env = env_factory(env_name)
  observation_spec = tf_env.observation_spec()
  action_spec = tf_env.action_spec()

  # Prepare data.
  full_data = get_offline_data(tf_env)

  # Split data.
  n_train = min(n_train, full_data.size)
  logging.info('n_train %s.', n_train)
  if use_seed_for_data:
    rand = np.random.RandomState(seed)
  else:
    rand = np.random.RandomState(0)
  shuffled_indices = utils.shuffle_indices_with_steps(
      n=full_data.size, steps=shuffle_steps, rand=rand)
  train_indices = shuffled_indices[:n_train]
  train_data = full_data.create_view(train_indices)

  # Create agent.
  agent_flags = utils.Flags(
      observation_spec=observation_spec,
      action_spec=action_spec,
      model_params=model_params,
      optimizers=optimizers,
      batch_size=batch_size,
      weight_decays=weight_decays,
      update_freq=update_freq,
      update_rate=update_rate,
      discount=discount,
      train_data=train_data)
  agent_args = agent_module.Config(agent_flags).agent_args
  my_agent_arg_dict = {}
  for k in vars(agent_args):
      my_agent_arg_dict[k] = vars(agent_args)[k]
  my_agent_arg_dict['behavior_ckpt_file'] = behavior_ckpt_file
  my_agent_arg_dict['value_penalty'] = value_penalty
  print('agent_args:', my_agent_arg_dict)
  #agent = agent_module.Agent(**vars(agent_args))
  agent = agent_module.Agent(**my_agent_arg_dict)
  agent_ckpt_name = os.path.join(log_dir, 'agent')

  # Restore agent from checkpoint if there exists one.
  if tf.io.gfile.exists('{}.index'.format(agent_ckpt_name)):
    logging.info('Checkpoint found at %s.', agent_ckpt_name)
    agent.restore(agent_ckpt_name)

  # Train agent.
  train_summary_dir = os.path.join(log_dir, 'train')
  eval_summary_dir = os.path.join(log_dir, 'eval')
  train_summary_writer = tf0.compat.v2.summary.create_file_writer(
      train_summary_dir)
  eval_summary_writers = collections.OrderedDict()
  for policy_key in agent.test_policies.keys():
    eval_summary_writer = tf0.compat.v2.summary.create_file_writer(
        os.path.join(eval_summary_dir, policy_key))
    eval_summary_writers[policy_key] = eval_summary_writer
  eval_results = []

  time_st_total = time.time()
  time_st = time.time()
  step = agent.global_step
  timed_at_step = step
  while step < total_train_steps:
    agent.train_step()
    step = agent.global_step
    if step % summary_freq == 0 or step == total_train_steps:
      agent.write_train_summary(train_summary_writer)
    if step % print_freq == 0 or step == total_train_steps:
      agent.print_train_info()
    if step % eval_freq == 0 or step == total_train_steps:
      time_ed = time.time()
      time_cost = time_ed - time_st
      logging.info(
          'Training at %.4g steps/s.', (step - timed_at_step) / time_cost)
      eval_result, eval_infos = train_eval_utils.eval_policies(
          tf_env, agent.test_policies, n_eval_episodes)
      eval_results.append([step] + eval_result)
      logging.info('Testing at step %d:', step)
      for policy_key, policy_info in eval_infos.items():
        logging.info(utils.get_summary_str(
            step=None, info=policy_info, prefix=policy_key+': '))
        utils.write_summary(eval_summary_writers[policy_key], step, policy_info)
      time_st = time.time()
      timed_at_step = step
    if step % save_freq == 0:
      agent.save(agent_ckpt_name)
      logging.info('Agent saved at %s.', agent_ckpt_name)

  agent.save(agent_ckpt_name)
  time_cost = time.time() - time_st_total
  logging.info('Training finished, time cost %.4gs.', time_cost)
  return np.array(eval_results)
