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
mounts.append(doodad.MountLocal(local_dir='/data/doodad_results/merge-random',
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



"""
ENV=walker2d-medium-v0
python scripts/brac/train_offline.py \
  --alsologtostderr --sub_dir=0 \
  --env_name=$ENV --identifier="train_bc" \
  --agent_name=bc \
  --total_train_steps=300000 \
  --n_train=1000000 \
  --gin_bindings="train_eval_offline.model_params=((200, 200),)" \
  --gin_bindings="train_eval_offline.batch_size=256" \
  --gin_bindings="train_eval_offline.optimizers=(('adam', 5e-4),)"
"""

doodad.run_python(
    target='brac/train_offline.py',
    mode=local_launcher,
    mounts=mounts,
    docker_image='justinfu/brac_flow:0.3',
    verbose=True,
    cli_args='--alsologtostderr --save_freq=1000 --sub_dir=0 --env_name=%s --identifier="train_bc" --agent_name=bc --total_train_steps=300000 --n_train=1000000 --model_arch=0 --opt_params=0' % env_name
             #'--gin_bindings="train_eval_offline.model_params=((200, 200),)" ' + \
             #'--gin_bindings="train_eval_offline.batch_size=256" ' + \
             #'--gin_bindings="train_eval_offline.optimizers=((\'adam\', 5e-4),)"'
)

