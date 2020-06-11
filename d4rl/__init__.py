import os
import sys

import d4rl.locomotion
import d4rl.hand_manipulation_suite
import d4rl.pointmaze
import d4rl.gym_minigrid
import d4rl.gym_mujoco
from d4rl.offline_env import set_dataset_path, get_keys

SUPPRESS_MESSAGES = bool(os.environ.get('D4RL_SUPPRESS_IMPORT_ERROR', 0))

_ERROR_MESSAGE = 'Warning: %s failed to import. Set the environment variable D4RL_SUPPRESS_IMPORT_ERROR=1 to suppress this message.'

try:
    import d4rl.flow
except ImportError as e:
    if not SUPPRESS_MESSAGES:
        print(_ERROR_MESSAGE % 'Flow', file=sys.stderr)
        print(e, file=sys.stderr)

try:
    import d4rl.kitchen
except ImportError as e:
    if not SUPPRESS_MESSAGES:
        print(_ERROR_MESSAGE % 'FrankaKitchen', file=sys.stderr)
        print(e, file=sys.stderr)

try:
    import d4rl.carla
except ImportError as e:
    if not SUPPRESS_MESSAGES:
        print(_ERROR_MESSAGE % 'CARLA', file=sys.stderr)
        print(e, file=sys.stderr)

