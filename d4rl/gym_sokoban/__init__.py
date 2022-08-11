import logging
import pkg_resources
import json
from gym.envs.registration import register

logger = logging.getLogger(__name__)

resource_package = __name__
env_json = pkg_resources.resource_filename(resource_package, '/'.join(('environments', 'available_envs.json')))

with open(env_json) as f:

    envs = json.load(f)

    for env in envs:
        register(
            id=env["id"],
            entry_point=env["entry_point"]
        )
        
        
register(
    id='sokoban-expert-v0',
    entry_point='d4rl.modular_rl.envs:OfflineModularRlEnv',
    kwargs={
        'game': f'Sokoban-v0',
        'ref_min_score': -20,
        'ref_max_score': 12,
    }
)
        
