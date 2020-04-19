import sys
import numpy as np
import gym
import gym.spaces

from d4rl.pointmaze.gridcraft.grid_spec import REWARD, REWARD2, REWARD3, REWARD4, WALL, LAVA, TILES, START, RENDER_DICT
from d4rl.pointmaze.gridcraft.utils import one_hot_to_flat, flat_to_one_hot

ACT_NOOP = 0
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_DICT = {
    ACT_NOOP: [0,0],
    ACT_UP: [0, -1],
    ACT_LEFT: [-1, 0],
    ACT_RIGHT: [+1, 0],
    ACT_DOWN: [0, +1]
}
ACT_TO_STR = {
    ACT_NOOP: 'NOOP',
    ACT_UP: 'UP',
    ACT_LEFT: 'LEFT',
    ACT_RIGHT: 'RIGHT',
    ACT_DOWN: 'DOWN'
}

class TransitionModel(object):
    def __init__(self, gridspec, eps=0.2):
        self.gs = gridspec
        self.eps = eps

    def get_aprobs(self, s, a):
        # TODO: could probably output a matrix over all states...
        legal_moves = self.__get_legal_moves(s)
        p = np.zeros(len(ACT_DICT))
        p[list(legal_moves)] = self.eps / (len(legal_moves))
        if a in legal_moves:
            p[a] += 1.0-self.eps
        else:
            #p = np.array([1.0,0,0,0,0])  # NOOP
            p[ACT_NOOP] += 1.0-self.eps
        return p

    def __get_legal_moves(self, s):
        xy = np.array(self.gs.idx_to_xy(s))
        moves = {move for move in ACT_DICT if not self.gs.out_of_bounds(xy+ACT_DICT[move])
                                             and self.gs[xy+ACT_DICT[move]] != WALL}
        moves.add(ACT_NOOP)
        return moves


class RewardFunction(object):
    def __init__(self, rew_map=None, default=0):
        if rew_map is None:
            rew_map = {
                REWARD: 1.0,
                REWARD2: 2.0,
                REWARD3: 4.0,
                REWARD4: 8.0,
                LAVA: -100.0,
            }
        self.default = default
        self.rew_map = rew_map

    def __call__(self, gridspec, s, a, ns):
        val = gridspec[gridspec.idx_to_xy(s)]
        if val in self.rew_map:
            return self.rew_map[val]
        return self.default


class GridEnv(gym.Env):
    def __init__(self, gridspec, 
                 tiles=TILES,
                 rew_fn=None,
                 teps=0.0, 
                 max_timesteps=None,
                 rew_map=None,
                 terminal_states=None,
                 default_rew=0):
        self.num_states = len(gridspec)
        self.num_actions = 5
        self._env_args = {'teps': teps, 'max_timesteps': max_timesteps}
        self.gs = gridspec
        self.model = TransitionModel(gridspec, eps=teps)
        self.terminal_states = terminal_states
        if rew_fn is None:
            rew_fn = RewardFunction(rew_map=rew_map, default=default_rew)
        self.rew_fn = rew_fn
        self.possible_tiles = tiles
        self.max_timesteps = max_timesteps
        self._timestep = 0
        self._true_q = None  # q_vals for debugging
        super(GridEnv, self).__init__()

    def get_transitions(self, s, a):
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        if tile_type == LAVA: # Lava gets you stuck
            return {s: 1.0}

        aprobs = self.model.get_aprobs(s, a)
        t_dict = {}
        for sa in range(5):
            if aprobs[sa] > 0:
                next_s = self.gs.idx_to_xy(s) + ACT_DICT[sa]
                next_s_idx = self.gs.xy_to_idx(next_s)
                t_dict[next_s_idx] = t_dict.get(next_s_idx, 0.0) + aprobs[sa]
        return t_dict


    def step_stateless(self, s, a, verbose=False):
        aprobs = self.model.get_aprobs(s, a)
        samp_a = np.random.choice(range(5), p=aprobs)

        next_s = self.gs.idx_to_xy(s) + ACT_DICT[samp_a]
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        if tile_type == LAVA: # Lava gets you stuck
            next_s = self.gs.idx_to_xy(s)

        next_s_idx = self.gs.xy_to_idx(next_s)
        rew = self.rew_fn(self.gs, s, samp_a, next_s_idx)

        if verbose:
            print('Act: %s. Act Executed: %s' % (ACT_TO_STR[a], ACT_TO_STR[samp_a]))
        return next_s_idx, rew

    def step(self, a, verbose=False):
        ns, r = self.step_stateless(self.__state, a, verbose=verbose)
        traj_infos = {}
        self.__state = ns
        obs = ns #flat_to_one_hot(ns, len(self.gs))

        done = False
        self._timestep += 1
        if self.max_timesteps is not None:
            if self._timestep >= self.max_timesteps:
                done = True
        return obs, r, done, traj_infos

    def reset(self):
        start_idxs = np.array(np.where(self.gs.spec == START)).T
        start_idx = start_idxs[np.random.randint(0, start_idxs.shape[0])]
        start_idx = self.gs.xy_to_idx(start_idx)
        self.__state =start_idx
        self._timestep = 0
        return start_idx #flat_to_one_hot(start_idx, len(self.gs))

    def render(self, close=False, ostream=sys.stdout):
        if close:
            return

        state = self.__state
        ostream.write('-'*(self.gs.width+2)+'\n')
        for h in range(self.gs.height):
            ostream.write('|')
            for w in range(self.gs.width):
                if self.gs.xy_to_idx((w,h)) == state:
                    ostream.write('*')
                else:
                    val = self.gs[w, h]
                    ostream.write(RENDER_DICT[val])
            ostream.write('|\n')
        ostream.write('-' * (self.gs.width + 2)+'\n')

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)

    @property
    def observation_space(self):
        dO = len(self.gs)
        #return gym.spaces.Box(0,1,shape=dO)
        return gym.spaces.Discrete(dO)

    def transition_matrix(self):
        """Constructs this environment's transition matrix.

        Returns:
          A dS x dA x dS array where the entry transition_matrix[s, a, ns]
          corrsponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        ds = self.num_states
        da = self.num_actions
        transition_matrix = np.zeros((ds, da, ds))
        for s in range(ds):
            for a in range(da):
                transitions = self.get_transitions(s,a)
                for next_s in transitions:
                    transition_matrix[s, a, next_s] = transitions[next_s]
        return transition_matrix

    def reward_matrix(self):
        """Constructs this environment's reward matrix.

        Returns:
          A dS x dA x dS numpy array where the entry reward_matrix[s, a, ns]
          reward given to an agent when transitioning into state ns after taking
          action s from state s.
        """
        ds = self.num_states
        da = self.num_actions
        rew_matrix = np.zeros((ds, da, ds))
        for s in range(ds):
            for a in range(da):
                for ns in range(ds):
                    rew_matrix[s, a, ns] = self.rew_fn(self.gs, s, a, ns)
        return rew_matrix
