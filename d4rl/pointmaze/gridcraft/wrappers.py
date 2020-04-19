import numpy as np
from d4rl.pointmaze.gridcraft.grid_env import REWARD, GridEnv
from d4rl.pointmaze.gridcraft.wrappers import ObsWrapper
from gym.spaces import Box


class GridObsWrapper(ObsWrapper):
    def __init__(self, env):
        super(GridObsWrapper, self).__init__(env)

    def render(self):
        self.env.render()



class EyesWrapper(ObsWrapper):
    def __init__(self, env, range=4, types=(REWARD,), angle_thresh=0.8):
        super(EyesWrapper, self).__init__(env)
        self.types = types
        self.range = range
        self.angle_thresh = angle_thresh

        eyes_low = np.ones(5*len(types))
        eyes_high = np.ones(5*len(types))
        low = np.r_[env.observation_space.low, eyes_low]
        high = np.r_[env.observation_space.high, eyes_high]
        self.__observation_space = Box(low, high)

    def wrap_obs(self, obs, info=None):
        gs = self.env.gs  # grid spec
        xy = gs.idx_to_xy(self.env.obs_to_state(obs))
        #xy = np.array([x, y])

        extra_obs = []
        for tile_type in self.types:
            idxs = gs.find(tile_type).astype(np.float32)  # N x 2
            # gather all idxs that are close
            diffs = idxs-np.expand_dims(xy, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            valid_idxs = np.where(dists <= self.range)[0]
            if len(valid_idxs) == 0:
                eye_data = np.array([0,0,0,0,0], dtype=np.float32)
            else:
                diffs = diffs[valid_idxs, :]
                dists = dists[valid_idxs]+1e-6
                cosines = diffs[:,0]/dists
                cosines = np.r_[cosines, 0]
                sines = diffs[:,1]/dists
                sines = np.r_[sines, 0]
                on_target = 0.0
                if np.any(dists<=1.0):
                    on_target = 1.0
                eye_data = np.abs(np.array([on_target, np.max(cosines), np.min(cosines), np.max(sines), np.min(sines)]))
                eye_data[np.where(eye_data<=self.angle_thresh)] = 0
            extra_obs.append(eye_data)
        extra_obs = np.concatenate(extra_obs)
        obs = np.r_[obs, extra_obs]
        #if np.any(np.isnan(obs)):
        #    import pdb; pdb.set_trace()
        return obs

    def unwrap_obs(self, obs, info=None):
        if len(obs.shape) == 1:
            return obs[:-5*len(self.types)]
        else:
            return obs[:,:-5*len(self.types)]

    @property
    def observation_space(self):
        return self.__observation_space


"""
class CoordinateWiseWrapper(GridObsWrapper):
    def __init__(self, env):
        assert isinstance(env, GridEnv)
        super(CoordinateWiseWrapper, self).__init__(env)
        self.gs = env.gs
        self.dO = self.gs.width+self.gs.height

        self.__observation_space = Box(0, 1, self.dO)

    def wrap_obs(self, obs, info=None):
        state = one_hot_to_flat(obs)
        xy = self.gs.idx_to_xy(state)
        x = flat_to_one_hot(xy[0], self.gs.width)
        y = flat_to_one_hot(xy[1], self.gs.height)
        obs = np.r_[x, y]
        return obs

    def unwrap_obs(self, obs, info=None):

        if len(obs.shape) == 1:
            x = obs[:self.gs.width]
            y = obs[self.gs.width:]
            x = one_hot_to_flat(x)
            y = one_hot_to_flat(y)
            state = self.gs.xy_to_idx(np.c_[x,y])
            return flat_to_one_hot(state, self.dO)
        else:
            raise NotImplementedError()
"""


class RandomObsWrapper(GridObsWrapper):
    def __init__(self, env, dO):
        assert isinstance(env, GridEnv)
        super(RandomObsWrapper, self).__init__(env)
        self.gs = env.gs
        self.dO = dO
        self.obs_matrix = np.random.randn(self.dO, len(self.gs))
        self.__observation_space = Box(np.min(self.obs_matrix), np.max(self.obs_matrix), 
            shape=(self.dO,), dtype=np.float32)

    def wrap_obs(self, obs, info=None):
        return np.inner(self.obs_matrix, obs)

    def unwrap_obs(self, obs, info=None):
        raise NotImplementedError()

