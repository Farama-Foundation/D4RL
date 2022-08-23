from gym.envs.registration import register
from .levels import *


REF_MIN_SCORE = {}

REF_MAX_SCORE = {}

DATASET_URLS = {}
ALL_ENVS = []

# list from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
for agent in ['1RoomS12',
              '1RoomS16',
              '1RoomS20',
              '1RoomS8',
              'ActionObjDoor',
              'BlockedUnlockPickup',
              'BossLevel',
              'FindObjS5',
              'FindObjS6',
              'FindObjS7',
              'GoToDoor',
              'GoToImpUnlock',
              'GoToLocal',
              'GoToObjDoor',
              'GoToObjMaze',
              'GoToObj',
              'GoToRedBallGrey',
              'GoToRedBall',
              'GoToSeq',
              'GoTo',
              'KeyCorridorS3R1',
              'KeyCorridorS3R2',
              'KeyCorridorS3R3',
              'KeyCorridorS4R3',
              'KeyCorridorS5R3',
              'KeyCorridorS6R3',
              'MoveTwoAcrossS5N2',
              'MoveTwoAcrossS8N9',
              'OpenDoorColor',
              'OpenDoorLoc',
              'OpenDoorsOrderN2',
              'OpenDoorsOrderN4',
              'OpenDoor',
              'OpenRedBlueDoors',
              'OpenRedDoor',
              'OpenTwoDoors',
              'Open',
              'PickupAbove',
              'PickupDist',
              'PickupLoc',
              'Pickup',
              'PutNextLocal',
              'PutNextS4N1',
              'PutNextS5N1',
              'PutNextS6N3',
              'PutNextS7N4',
              'PutNext',
              'SynthLoc',
              'SynthSeq',
              'Synth',
              'UnblockPickup',
              'UnlockPickupDist',
              'UnlockPickup',
              'UnlockToUnlock',
              'Unlock']:
    env_name = '%s-expert-v0' % agent
    ALL_ENVS.append(env_name)
    register(
        id=env_name,
        entry_point='d4rl.babyai.envs:OfflineBabyaiEnv',
        kwargs={
            'game': f'BabyAI-{agent}-v0',
            'ref_min_score': 0,
            'ref_max_score': 1,
        }
    )