import os
import glob
from gym.envs.registration import register
REF_MAX_SCORE={
    'bin-picking-v2': 4292.8,
    'button-press-topdown-v2': 3888.4,
    'button-press-topdown-wall-v2': 3881.9,
    'button-press-v2': 3623.3,
    'button-press-wall-v2': 3675.1,
    'coffee-button-v2': 4261.5,
    'coffee-pull-v2': 4201.5,
    'coffee-push-v2': 4184.8,
    'dial-turn-v2': 4658.2,
    'door-close-v2': 4540.0,
    'door-lock-v2': 3913.5,
    'door-open-v2': 4561.0,
    'door-unlock-v2': 4625.9,
    'drawer-close-v2': 4871.3,
    'drawer-open-v2': 4224.6,
    'faucet-close-v2': 4757.6,
    'faucet-open-v2': 4774.3,
    'hammer-v2': 4613.9,
    'hand-insert-v2': 4544.1,
    'handle-press-side-v2': 4844.2,
    'handle-press-v2': 4867.2,
    'handle-pull-side-v2': 4635.5,
    'handle-pull-v2': 4492.6,
    'peg-insert-side-v2': 4605.7,
    'peg-unplug-side-v2': 4464.6,
    'pick-place-v2': 4428.6,
    'plate-slide-back-side-v2': 4788.9,
    'plate-slide-back-v2': 4777.9,
    'plate-slide-side-v2': 4683.2,
    'plate-slide-v2': 4674.1,
    'push-back-v2': 0,
    'push-v2': 4750.8,
    'push-wall-v2': 4676.8,
    'reach-v2': 4863.4,
    'reach-wall-v2': 4806.4,
    'soccer-v2': 4463.7,
    'stick-pull-v2': 4228.8,
    'sweep-into-v2': 4631.5,
    'sweep-v2': 4487.1,
    'window-close-v2': 4563.4,
    'window-open-v2': 4406.4,
    
}
env_names = glob.glob('/nfs/dgx08/home/lcy/metaworld/garage/datasets/*.hdf5')

env_names = [os.path.basename(env_name)[:-8] for env_name in env_names]

# list from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
for agent in env_names:
    offline_env_name = '%s-expert-v0' % agent
    register(
        id=offline_env_name,
        entry_point='d4rl.metaworld.envs:OfflineMWRlEnv',
        kwargs={
            'game': f'{agent}-v2',
            'ref_min_score': 0,
            'ref_max_score': REF_MAX_SCORE[f'{agent}-v2'],
        }
    )

ALL_ENVS = env_names