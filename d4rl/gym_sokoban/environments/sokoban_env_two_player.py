from .sokoban_env import SokobanEnv, CHANGE_COORDINATES
from gym.spaces import Box
from gym.spaces.discrete import Discrete
from .render_utils import room_to_rgb, room_to_tiny_world_rgb, color_player_two, color_tiny_player_two
import numpy as np


class TwoPlayerSokobanEnv(SokobanEnv):

    def __init__(self,
             dim_room=(10, 10),
             max_steps=120,
             num_boxes=3,
             num_gen_steps=None):
        
        super(TwoPlayerSokobanEnv, self).__init__(dim_room, max_steps, num_boxes, num_gen_steps, reset=False)
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        self.boxes_are_on_target = [False] * num_boxes
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.player_position = []
        self.player_positions = {0: [0,0], 1: [1,1]}

        _ = self.reset(second_player=True)

    def reset(self, render_mode='rgb_array',second_player=True):
        super(TwoPlayerSokobanEnv, self).reset(second_player=second_player)

        self.player_positions = {
            0: np.argwhere(self.room_state == 5)[0],
            1: np.argwhere(self.room_state == 5)[1]
        }

        return self.render(mode=render_mode)

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        active_player = 0
        if action > 8:
            active_player = 1

        self.player_position = self.player_positions[active_player]

        player_action = (action-1) % 8

        if action == 0:
            moved_player = False
            moved_box = False
            active_player = -1

        # All push actions are in the range of [0, 3]
        elif player_action < 4:
            moved_player, moved_box = self._push(player_action + 1)

        elif player_action < 8:
            moved_player = self._move(player_action + 1)
            moved_box = False

        self.player_positions[active_player] = self.player_position

        self._calc_reward()

        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
            "action,active_player": active_player
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def get_image(self, mode, scale=1):

        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
            img = color_tiny_player_two(img, self.player_positions[1], self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)
            img = color_player_two(img, self.player_positions[1], self.room_fixed)

        return img

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'P1: push up',
    2: 'P1: push down',
    3: 'P1: push left',
    4: 'P1: push right',
    5: 'P1: move up',
    6: 'P1: move down',
    7: 'P1: move left',
    8: 'P1: move right',
    9: 'P2: push up',
    10: 'P2: push down',
    11: 'P2: push left',
    12: 'P2: push right',
    13: 'P2: move up',
    14: 'P2: move down',
    15: 'P2: move left',
    16: 'P2: move right'
}

