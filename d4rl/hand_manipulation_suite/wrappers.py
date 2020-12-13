import gym
import numpy as np

from d4rl.hand_manipulation_suite.door_v0 import DoorEnvV0
from d4rl.hand_manipulation_suite.hammer_v0 import HammerEnvV0
from d4rl.hand_manipulation_suite.pen_v0 import PenEnvV0
from d4rl.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

class MJCVisionWrapper(gym.ObservationWrapper):
    def __init__(self, env, image_size=84, jpg=False, jpg_quality=95):
        super(MJCVisionWrapper, self).__init__(env)
        self.image_size = image_size
        env.offscreen_viewer_setup()
        self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(image_size, image_size, 3),
                dtype=np.float32)
        self.jpg = jpg
        self.jpg_quality = jpg_quality

    def observation(self, s):
        image = self.env.sim.render(width=self.image_size, height=self.image_size,
                                     mode='offscreen', camera_name=None, device_id=0)
        image = image[::-1,:,:]
        image = image / 255.0
        if self.jpg:
            image = img_utils.encode_jpg(image, quality=self.jpg_quality)
        return image

def get_door_vision_env(**kwargs):
    return MJCVisionWrapper(DoorEnvV0(**kwargs), jpg=True)

def get_relocate_vision_env(**kwargs):
    return MJCVisionWrapper(RelocateEnvV0(**kwargs), jpg=True)

def get_hammer_vision_env(**kwargs):
    return MJCVisionWrapper(HammerEnvV0(**kwargs), jpg=True)

def get_pen_vision_env(**kwargs):
    return MJCVisionWrapper(PenEnvV0(**kwargs), jpg=True)

