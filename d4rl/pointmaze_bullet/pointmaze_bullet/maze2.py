import os
import hashlib
import .bullet_env
from pybullet_envs import env_bases
from pybullet_envs import scene_abstract

from d4rl.pointmaze import maze_model
from d4rl import offline_env

class MazeRobot(bullet_env.MJCFBaseRobot):
    def __init__(self, maze_spec):
        model = maze_model.point_maze(maze_spec)
        maze_hash = hashlib.md5(maze_spec).hexdigest()
        filename = os.path.join(offline_env.DATASET_PATH, 'bullet_xml', maze_hash)
        if not os.path.exists(f):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with model.asfile() as f:
                model_xml = f.read()
            with open(filename, 'w') as f:
                f.write(model_xml)
        super(MazeRobot, self).__init__(self, model_xml=filename,
                                              robot_name='maze2d',
                                              action_dim=2,
                                              observation_dim=4)

    def calc_state(self):
        import pdb; pdb.set_trace()
        pass


class Maze2DBulletEnv(env_bases.MJCFBaseBulletEnv):

    def __init__(self, maze_spec, 
                 reward_type='dense',
                 reset_target=False):
        self.robot = MazeRobot()
        env_bases.MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()

        self._target = np.array([0.0,0.0])

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            # If no goal, use the first empty tile
            self.set_target(np.array(self.reset_locations[0]).astype(self.observation_space.dtype))
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations

    def create_single_player_scene(self, bullet_client):
        return scene_abstract.SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self):
        if (self.stateId >= 0):
          self._p.restoreState(self.stateId)
        r = env_bases.MJCFBaseBulletEnv.reset(self)
        if (self.stateId < 0):
          self.stateId = self._p.saveState()
        return r

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        #self.do_simulation(action, self.frame_skip)
        self.robot.apply_action(action)
        self.scene.global_step()
        #self.set_marker()
        ob = self.robot.calc_state()
        if self.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
        done = False
        self.HUD(state, action, done)
        return ob, reward, done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
            target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        self._target = target_location

    def clip_velocity(self):
        print('clip velocity not implemented')
        pass
        #qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        #self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        idx = self.np_random.choice(len(self.empty_and_goal_locations))
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()
