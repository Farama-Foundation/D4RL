from gym.envs.registration import register
import os
import glob

ENV_DIR = 'd4rl/modular_rl/environments'
XML_DIR = 'd4rl/modular_rl/environments/xmls'

def register_env(env_name, max_episode_steps=1000):
    """register the MuJoCo envs with Gym and return the per-limb observation size and max action value (for modular
    policy training) """
    # existing envs
    xml = os.path.join(XML_DIR, "{}.xml".format(env_name))

    env_name = os.path.basename(xml)[:-4]
    env_file = env_name
    # create a copy of modular environment for custom xml model
    if not os.path.exists(os.path.join(ENV_DIR, '{}.py'.format(env_name))):
        raise Exception('Please create a copy of the modular environment file for each custom environment.')
    params = {'xml': os.path.abspath(xml)}
    # register with gym
    register(id=("%s-v0" % env_name),
             max_episode_steps=max_episode_steps,
             entry_point="d4rl.modular_rl.environments.%s:ModularEnv" % env_file,
             kwargs=params)
    return

REF_MIN_SCORE = {}

REF_MAX_SCORE = {
    'cheetah_2_back': 1051.44,
    'cheetah_7_full': 6716.04,
    'cheetah_4_back': 4360.56,
    'humanoid_2d_8_right_knee': 4698.78,
    'walker_3_flipped': 3705,
    'cheetah_5_back': 5737,
    'walker_5_main': 7280,
    'walker_2_flipped':  3337.81,
    'walker_2_main': 3492.04,
    'cheetah_2_front': 2906.03,
    'walker_6_flipped': 5299.01,
    'hopper_3': 4023.26,
    'walker_6_main': 5543.99,
    'walker_3_main': 3747.62,
    'humanoid_2d_9_full': 7241.81,
    'cheetah_3_front': 3861.28,
    'walker_4_flipped': 5200.64,
    'cheetah_4_allfront': 4528.67,
    'cheetah_3_back': 2384.70,
    'walker_4_main': 4838.98,
    'humanoid_2d_7_right_arm': 7106.11,
    'cheetah_3_balanced': 2126.00,
    'walker_7_main': 5560.54,
    'hopper_5': 5185.95,
    'humanoid_2d_7_left_arm': 7068.89,
    'walker_5_flipped': 6998.63,
    'cheetah_5_balanced': 6539.41,
    'cheetah_6_back': 7047.49,
    'cheetah_5_front': 4052.479,
    'humanoid_2d_7_left_leg': 3668.63,
    'humanoid_2d_7_lower_arms': 7285.21,
    'cheetah_4_allback': 3364.36,
    'cheetah_6_front': 5489.42,
    'walker_7_flipped': 5400.76,
    'cheetah_4_front': 4374.95,
    'humanoid_2d_7_right_leg': 3584.78,
    'hopper_4': 4687.14,
    'humanoid_2d_8_left_knee': 4521.38
}


env_names = glob.glob('/nfs/dgx08/raid/modular_rl/*.hdf5')

env_names = [os.path.basename(env_name)[:-5] for env_name in env_names]

# list from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
for agent in env_names:
    register_env(agent)
    offline_env_name = '%s-expert-v0' % agent
    register(
        id=offline_env_name,
        entry_point='d4rl.modular_rl.envs:OfflineModularRlEnv',
        kwargs={
            'game': f'{agent}-v0',
            'ref_min_score': 0,
            'ref_max_score': REF_MAX_SCORE[agent],
        }
    )

ALL_ENVS = env_names