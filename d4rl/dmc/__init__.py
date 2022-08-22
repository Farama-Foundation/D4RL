from gym.envs.registration import register, registry
from .dataset_info import DATASET_URLS, REF_MAX_SCORE, REF_MIN_SCORE


for domain_and_task, uri in DATASET_URLS.items():
    env_id = f'dmc-{domain_and_task}-expert-v1'
    register(
        id=env_id,
        entry_point='d4rl.dmc.env:OfflineDMCEnv',
        max_episode_steps=1000,
        kwargs={
            'domain_name': domain_and_task.split("-")[0],
            'task_name': domain_and_task.split("-")[1],
            'from_pixels': False,
            'dataset_url': uri,
            'ref_min_score': REF_MIN_SCORE[domain_and_task],
            'ref_max_score': REF_MAX_SCORE[domain_and_task]
        }
    )