import random

import numpy as np

from d4rl.pointmaze import q_iteration
from d4rl.pointmaze.gridcraft import grid_env, grid_spec

MAZE = (
    "###################\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOOOOOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "####O#########O####\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "#OOOOOOOOOOOOOOOOO#\\"
    + "#OOOOOOOO#OOOOOOOO#\\"
    + "###################\\"
)


# NLUDR -> RDLU
TRANSLATE_DIRECTION = {
    0: None,
    1: 3,  # 3,
    2: 1,  # 1,
    3: 2,  # 2,
    4: 0,  # 0,
}

RIGHT = 1
LEFT = 0
FORWARD = 2


class FourRoomController:
    def __init__(self):
        self.env = grid_env.GridEnv(grid_spec.spec_from_string(MAZE))
        self.reset_locations = list(zip(*np.where(self.env.gs.spec == grid_spec.EMPTY)))

    def sample_target(self):
        return random.choice(self.reset_locations)

    def set_target(self, target):
        self.target = target
        self.env.gs[target] = grid_spec.REWARD
        self.q_values = q_iteration.q_iteration(
            env=self.env, num_itrs=32, discount=0.99
        )
        self.env.gs[target] = grid_spec.EMPTY

    def get_action(self, pos, orientation):
        if tuple(pos) == tuple(self.target):
            done = True
        else:
            done = False
        env_pos_idx = self.env.gs.xy_to_idx(pos)
        qvalues = self.q_values[env_pos_idx]
        direction = TRANSLATE_DIRECTION[np.argmax(qvalues)]
        # tgt_pos, _ = self.env.step_stateless(env_pos_idx, np.argmax(qvalues))
        # tgt_pos = self.env.gs.idx_to_xy(tgt_pos)
        # print('\tcmd_dir:', direction, np.argmax(qvalues), qvalues, tgt_pos)
        # infos = {}
        # infos['tgt_pos'] = tgt_pos
        if orientation == direction or direction is None:
            return FORWARD, done
        else:
            return get_turn(orientation, direction), done


# RDLU
TURN_DIRS = [
    [None, RIGHT, RIGHT, LEFT],  # R
    [LEFT, None, RIGHT, RIGHT],  # D
    [RIGHT, LEFT, None, RIGHT],  # L
    [RIGHT, RIGHT, LEFT, None],  # U
]


def get_turn(ori, tgt_ori):
    return TURN_DIRS[ori][tgt_ori]
