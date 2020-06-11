import os
import doodad
from doodad.wrappers.sweeper import launcher
import d4rl
import d4rl.flow
import gym

#env_name = 'maze2d-eval-umaze-v1'
env_name = 'flow-merge-random-v0'
env = gym.make(env_name)
_, dataset = os.path.split(env.dataset_filepath)
dirname, _ = os.path.splitext(dataset)

mounts = []
#mounts.append(doodad.MountLocal(local_dir='~/code/batch_rl_aviral',
#                              mount_point='/code/batch_rl_private', pythonpath=True))
mounts.append(doodad.MountLocal(local_dir='~/code/d4rl',
                              mount_point='/code/d4rl', pythonpath=True))
#mounts.append(doodad.MountLocal(local_dir='~/.d4rl/rlkit/%s' % dirname,
#                              mount_point='/datasets'))
mounts.append(doodad.MountLocal(local_dir='/data/doodad_results',
                                mount_point='/root/tmp/offlinerl', output=True))

gcp_launcher = doodad.GCPMode(
    gcp_bucket='justin-doodad',
    gcp_log_path='doodad/logs/bear',
    gcp_project='qlearning000',
    instance_type='n1-standard-1',
    zone='us-west1-a',
    gcp_image='ubuntu-1804-docker-gpu',
    gcp_image_project='qlearning000'
)
local_launcher = doodad.LocalMode()

#!/bin/bash
#ALPHA=1.0
#PLR=3e-05
#VALUE_PENALTY=False
#DIVERGENCE=kl
#ENV=walker2d-medium-v0
#DATA=example
#B_CKPT=$HOME/tmp/offlinerl/learn/$ENV/train_bc/bc/0/0/agent_behavior
#python train_offline.py \
#  --alsologtostderr --sub_dir=auto \
#  --env_name=$ENV \
#  --agent_name=brac_primal \
#  --total_train_steps=500000 \
#  --gin_bindings="brac_primal_agent.Agent.behavior_ckpt_file='$B_CKPT'" \
#  --gin_bindings="brac_primal_agent.Agent.alpha=$ALPHA" \
#  --gin_bindings="brac_primal_agent.Agent.value_penalty=$VALUE_PENALTY" \
#  --gin_bindings="brac_primal_agent.Agent.divergence_name='$DIVERGENCE'" \
#  --gin_bindings="train_eval_offline.model_params=(((300, 300), (200, 200),), 2)" \
#  --gin_bindings="train_eval_offline.batch_size=256" \
#  --gin_bindings="train_eval_offline.optimizers=(('adam', 1e-3), ('adam', $PLR), ('adam', 1e-3))" \

# Find b_ckpt_file
# /data/brac_flow/2020_05_20_brac_flow/outputs/exp_logs/learn/flow-ring-random-v0/train_bc/bc/0/1/
mounts.append(doodad.MountLocal(local_dir='/data/brac_flow/2020_05_20_brac_flow/outputs/exp_logs/learn/flow-merge-random-v0/train_bc/bc/0/0',
                                mount_point='/data/b_ckpt'
                                ))

#mounts.append(doodad.MountLocal(local_dir='/data/doodad_results/merge-random/learn/flow-merge-random-v0/train_bc/bc/0/0/',
#                                mount_point='/data/b_ckpt'
#                                ))

cli_args = '--alsologtostderr --sub_dir=auto --env_name={env_name} --agent_name={agent_type} --total_train_steps=500000 --model_arch=1 --opt_params=1 ' + \
        '--b_ckpt={b_ckpt} --value_penalty={value_penalty} ' 
cli_args = cli_args.format(
        env_name=env_name,
        agent_type='brac_primal',
        alpha=1.0,
        value_penalty=0,
        divergence='kl',
        b_ckpt='/data/b_ckpt/agent_behavior'
    )

doodad.run_python(
    target='brac/train_offline.py',
    mode=local_launcher,
    mounts=mounts,
    docker_image='justinfu/brac_flow:0.3',
    verbose=True,
    cli_args=cli_args
)

