from .kitchen_envs import KitchenMicrowaveKettleLightSliderV0
from gym.envs.registration import register

register(
    id='kitchen_microwave_kettle_light_slider-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        #TODO: 'dataset_url':
    }
)
