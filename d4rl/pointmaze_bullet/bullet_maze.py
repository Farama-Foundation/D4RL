import os
import hashlib
import numpy as np
from pybullet_envs import env_bases
from pybullet_envs import scene_abstract

from d4rl.pointmaze_bullet import bullet_robot
from d4rl.pointmaze import maze_model
from d4rl import offline_env

class MazeRobot(bullet_robot.MJCFBasedRobot):
    def __init__(self, maze_spec):
        model = maze_model.point_maze(maze_spec)
        maze_hash = hashlib.md5(maze_spec.encode('ascii')).hexdigest()
        filename = os.path.join(offline_env.DATASET_PATH, 'tmp_bullet_xml', maze_hash+'.xml')
        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with model.asfile() as f:
                model_xml = f.read()
            with open(filename, 'w') as f:
                f.write(model_xml)

        self.dt = 0.0165
        self.last_qpos = None
        super(MazeRobot, self).__init__(model_xml=filename,
                                        robot_name='maze2d',
                                        action_dim=2,
                                        obs_dim=4,
                                        self_collision=True)
    @property
    def qpos(self):
        x = self.particle.get_position()[0:2]
        return x

    @property
    def qvel(self):
        #vx = self.particle.speed()[0:2]
        #vx = np.array([self.ball_x.get_velocity(), self.ball_y.get_velocity()], dtype=np.float32)
        vx = (self.qpos - self.last_qpos) / self.dt
        return vx

    def calc_state(self):
        #import pdb; pdb.set_trace()
        return np.concatenate([self.qpos - 1.0, self.qvel])

    def set_state(self, qpos, qvel):
        self.particle.reset_position(np.array([qpos[0], qpos[1], 0.0]))
        self.particle.reset_velocity(np.array([qvel[0], qvel[1], 0.0]))
        self.last_qpos = self.qpos
        #self.ball_x.set_velocity(qvel[0])
        #self.ball_y.set_velocity(qvel[1])

    def get_obs(self):
        return self.calc_state()

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.particle = self.parts["particle"]
        self.ball_x = self.jdict["ball_x"]
        self.ball_y = self.jdict["ball_y"]
        #u = self.np_random.uniform(low=-.1, high=.1)
        #self.j1.reset_current_position(u if not self.swingup else 3.1415 + u, 0)
        self.ball_x.set_motor_torque(0)
        self.ball_y.set_motor_torque(0)
        self.last_qpos = self.qpos

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.last_qpos = self.qpos
        self.ball_x.set_motor_torque(a[0]*10)
        self.ball_y.set_motor_torque(a[1]*10)


class Maze2DBulletEnv(env_bases.MJCFBaseBulletEnv, offline_env.OfflineEnv):

    def __init__(self, maze_spec, 
                 reward_type='dense',
                 reset_target=False,
                 **kwargs):
        self.robot = MazeRobot(maze_spec)
        env_bases.MJCFBaseBulletEnv.__init__(self, self.robot)
        offline_env.OfflineEnv.__init__(self, **kwargs)
        self.stateId = -1

        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = maze_model.parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == maze_model.EMPTY)))
        self.reset_locations.sort()

        self._target = np.array([0.0,0.0])

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == maze_model.GOAL)))
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

        self.reset_model()
        ob = self.robot.calc_state()
        return ob

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        #self.clip_velocity()
        self.robot.apply_action(action)
        self.scene.global_step()
        ob = self.robot.calc_state()
        if self.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
        done = False
        self.HUD(ob, action, done)
        return ob, reward, done, {}

    def camera_adjust(self):
        qpos = self.robot.qpos
        x = qpos[0]
        y = qpos[1]
        self.camera.move_and_look_at(x, y, 1.4, x, y, 1.0)

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
            target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=2)
        self._target = target_location

    def clip_velocity(self):
        qvel = np.clip(self.robot.qvel, -5.0, 5.0)
        self.robot.set_state(self.robot.qpos, qvel)

    def reset_model(self):
        idx = self.np_random.choice(len(self.empty_and_goal_locations))
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=2)
        qvel = self.np_random.randn(2) * .1
        self.robot.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self.robot.get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=2)
        qvel = self.np_random.randn(2) * .1
        self.robot.set_state(qpos, qvel)
        return self.robot.get_obs()

