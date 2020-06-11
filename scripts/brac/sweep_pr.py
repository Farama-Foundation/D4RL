import os
import doodad
from doodad.wrappers.sweeper import launcher
import d4rl
import d4rl.flow
import gym

ENVS = [
    #'flow-ring-random-v0',
    #'flow-ring-controller-v0',
    'flow-merge-random-v0',
    'flow-merge-controller-v0',
]

mounts = []
mounts.append(doodad.MountLocal(local_dir='~/code/d4rl',
                              mount_point='/code/d4rl', pythonpath=True))


sweeper = launcher.DoodadSweeper(
    mounts=mounts,
    docker_img='justinfu/brac_flow:0.3',
    docker_output_dir='/root/tmp/offlinerl',
    gcp_bucket_name='justin-doodad',
    gcp_image='ubuntu-1804-docker-gpu',
    gcp_project='qlearning000'
)


for env_name in ENVS:
    env = gym.make(env_name)
    _, dataset = os.path.split(env.dataset_filepath)
    dirname, _ = os.path.splitext(dataset)


    params = {
        'env_name': [env_name],
        'seed': range(3),
        'agent_name': ['brac_primal'],
        'total_train_steps': [100000],
        'sub_dir': ['auto'],
        'model_arch': [1],
        'opt_params': [1],
        'value_penalty': [0],
        'b_ckpt': ['/data/b_ckpt/agent_behavior']
    }
    # --alsologtostderr --sub_dir=0 --env_name=%s --identifier="train_bc" --agent_name=bc --total_train_steps=300000 --n_train=1000000 ' % env_name

    #data_mount = doodad.MountLocal(local_dir='~/.d4rl/rlkit/%s' % dirname,
    #                          mount_point='/datasets')
    behavior_mount = doodad.MountLocal(local_dir='/data/brac_flow/2020_05_20_brac_flow/outputs/exp_logs/learn/%s/train_bc/bc/0/1' % env_name,
                                    mount_point='/data/b_ckpt'
                                    )
    sweeper.run_sweep_gcp(
        target='brac/train_offline.py',
        region='us-west1-a',
        params=params,
        extra_mounts=[behavior_mount],
        instance_type='n1-standard-4',
        log_prefix='brac_flow_pr2'
    )

