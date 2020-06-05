from .carla_env import CarlaObsDictEnv
from .carla_env import CarlaObsEnv
from gym.envs.registration import register

# Smaller dataset with only positive demonstrations.
register(
    id='carla-lane-v0',
    entry_point='d4rl.carla:CarlaObsEnv',
    max_episode_steps=250,
    kwargs={
        'ref_min_score': -0.8503839912088142,
        'ref_max_score': 1023.5784385429523, 
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_lane_follow_flat-v0.hdf5',
        'reward_type': 'lane_follow',
        'carla_args': dict(
            vision_size=48,
            vision_fov=48,
            weather=False,
            frame_skip=1,
            steps=250,
            multiagent=True,
            lane=0,
            lights=False,
            record_dir="None",
        )
    }
)

register(
    id='carla-lane-imgobs-v0',
    entry_point='d4rl.carla:CarlaObsDictEnv',
    max_episode_steps=250,
    kwargs={
        'ref_min_score': -0.8503839912088142,
        'ref_max_score': 1023.5784385429523, 
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_lane_follow-v0.hdf5',
        'reward_type': 'lane_follow',
        'carla_args': dict(
            vision_size=48,
            vision_fov=48,
            weather=False,
            frame_skip=1,
            steps=250,
            multiagent=True,
            lane=0,
            lights=False,
            record_dir="None",
        )
    }
)

register(
    id='carla-lane-render-v0',
    entry_point='d4rl.carla:CarlaDictEnv',
    max_episode_steps=250,
    kwargs={
        'ref_min_score': -0.8503839912088142,
        'ref_max_score': 1023.5784385429523, 
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_lane_follow-v0.hdf5',
        'reward_type': 'lane_follow',
        'render_images': True,
        'carla_args': dict(
            vision_size=48,
            vision_fov=48,
            weather=False,
            frame_skip=1,
            steps=250,
            multiagent=True,
            lane=0,
            lights=False,
            record_dir="None",
        )
    }
)


TOWN_STEPS = 1000
register(
    id='carla-town-v0',
    entry_point='d4rl.carla:CarlaObsEnv',
    max_episode_steps=TOWN_STEPS,
    kwargs={
        'ref_min_score': -122.228455,
        'ref_max_score': -64.62967840318221,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_subsamp_flat-v0.hdf5',
        'reward_type': 'goal_reaching',
        'carla_args': dict(
            vision_size=48,
            vision_fov=48,
            weather=False,
            frame_skip=1,
            steps=TOWN_STEPS,
            multiagent=True,
            lane=0,
            lights=False,
            record_dir="None",
        )
    }
)

register(
    id='carla-town-full-v0',
    entry_point='d4rl.carla:CarlaObsEnv',
    max_episode_steps=TOWN_STEPS,
    kwargs={
        'ref_min_score': -122.228455,
        'ref_max_score': -64.62967840318221,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_flat-v0.hdf5',
        'reward_type': 'goal_reaching',
        'carla_args': dict(
            vision_size=48,
            vision_fov=48,
            weather=False,
            frame_skip=1,
            steps=TOWN_STEPS,
            multiagent=True,
            lane=0,
            lights=False,
            record_dir="None",
        )
    }
)


register(
    id='carla-town-imgobs-v0',
    entry_point='d4rl.carla:CarlaObsDictEnv',
    max_episode_steps=TOWN_STEPS,
    kwargs={
        'ref_min_score': -122.228455,
        'ref_max_score': -64.62967840318221,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_subsamp-v0.hdf5',
        'reward_type': 'goal_reaching',
        'carla_args': dict(
            vision_size=48,
            vision_fov=48,
            weather=False,
            frame_skip=1,
            steps=TOWN_STEPS,
            multiagent=True,
            lane=0,
            lights=False,
            record_dir="None",
        )
    }
)

register(
    id='carla-town-imgobs-full-v0',
    entry_point='d4rl.carla:CarlaObsDictEnv',
    max_episode_steps=TOWN_STEPS,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town-v0.hdf5',
        'reward_type': 'goal_reaching',
        'carla_args': dict(
            vision_size=48,
            vision_fov=48,
            weather=False,
            frame_skip=1,
            steps=TOWN_STEPS,
            multiagent=True,
            lane=0,
            lights=False,
            record_dir="None",
        )
    }
)


register(
    id='carla-town-render-v0',
    entry_point='d4rl.carla:CarlaObsEnv',
    max_episode_steps=TOWN_STEPS,
    kwargs={
        'ref_min_score': -122.228455,
        'ref_max_score': -64.62967840318221,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_flat-v0.hdf5',
        'render_images': True,
        'reward_type': 'goal_reaching',
        'carla_args': dict(
            vision_size=48,
            vision_fov=48,
            weather=False,
            frame_skip=1,
            steps=TOWN_STEPS,
            multiagent=True,
            lane=0,
            lights=False,
            record_dir="None",
        )
    }
)
