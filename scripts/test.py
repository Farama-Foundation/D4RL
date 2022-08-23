import d4rl
import gym
import numpy as np

env = gym.make('PickupDist-expert-v0')

dataset = env.get_dataset()
# print('1')
obs = env.reset()
for _ in range(10):
    obs = env.step(0)
    print(obs)
# reward = dataset['rewards']
# terminal = dataset['terminals']
# done_idxs = []
# for i in range(len(terminal)):
#     if terminal[i]:
#         done_idxs.append(i)

# done_idxs = [-1] + done_idxs

# episode_reward = []

# for i in range(len(done_idxs)-1):
#     episode_reward.append(np.sum(reward[done_idxs[i]+1:done_idxs[i+1]+1]))

# print(np.max(episode_reward))
# print(np.min(episode_reward))


# for agent in ['1RoomS12',
#               '1RoomS16',
#               '1RoomS20',
#               '1RoomS8',
#               'ActionObjDoor',
#               'BlockedUnlockPickup',
#               'BossLevel',
#               'FindObjS5',
#               'FindObjS6',
#               'FindObjS7',
#               'GoToDoor',
#               'GoToImpUnlock',
#               'GoToLocal',
#               'GoToObjDoor',
#               'GoToObjMaze',
#               'GoToObj',
#               'GoToRedBallGrey',
#               'GoToRedBall',
#               'GoToSeq',
#               'GoTo',
#               'KeyCorridorS3R1',
#               'KeyCorridorS3R2',
#               'KeyCorridorS3R3',
#               'KeyCorridorS4R3',
#               'KeyCorridorS5R3',
#               'KeyCorridorS6R3',
#               'MoveTwoAcrossS5N2',
#               'MoveTwoAcrossS8N9',
#               'OpenDoorColor',
#               'OpenDoorLoc',
#               'OpenDoorsOrderN2',
#               'OpenDoorsOrderN4',
#               'OpenDoor',
#               'OpenRedBlueDoors',
#               'OpenRedDoor',
#               'OpenTwoDoors',
#               'Open',
#               'PickupAbove',
#               'PickupDist',
#               'PickupLoc',
#               'Pickup',
#               'PutNextLocal',
#               'PutNextS4N1',
#               'PutNextS5N1',
#               'PutNextS6N3',
#               'PutNextS7N4',
#               'PutNext',
#               'SynthLoc',
#               'SynthSeq',
#               'Synth',
#               'UnblockPickup',
#               'UnlockPickupDist',
#               'UnlockPickup',
#               'UnlockToUnlock',
#               'Unlock']