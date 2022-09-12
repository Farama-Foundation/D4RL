import numpy as np

from d4rl.pointmaze import q_iteration
from d4rl.pointmaze.gridcraft import grid_env, grid_spec

ZEROS = np.zeros((2,), dtype=np.float32)
ONES = np.zeros((2,), dtype=np.float32)


class WaypointController:
    def __init__(self, maze_str, solve_thresh=0.1, p_gain=10.0, d_gain=-1.0):
        self.maze_str = maze_str
        self._target = -1000 * ONES

        self.p_gain = p_gain
        self.d_gain = d_gain
        self.solve_thresh = solve_thresh
        self.vel_thresh = 0.1

        self._waypoint_idx = 0
        self._waypoints = []
        self._waypoint_prev_loc = ZEROS

        self.env = grid_env.GridEnv(grid_spec.spec_from_string(maze_str))

    def current_waypoint(self):
        return self._waypoints[self._waypoint_idx]

    def get_action(self, location, velocity, target):
        if np.linalg.norm(self._target - np.array(self.gridify_state(target))) > 1e-3:
            # print('New target!', target, 'old:', self._target)
            self._new_target(location, target)

        dist = np.linalg.norm(location - self._target)
        vel = self._waypoint_prev_loc - location
        vel_norm = np.linalg.norm(vel)
        task_not_solved = (dist >= self.solve_thresh) or (vel_norm >= self.vel_thresh)

        if task_not_solved:
            next_wpnt = self._waypoints[self._waypoint_idx]
        else:
            next_wpnt = self._target

        # Compute control
        prop = next_wpnt - location
        action = self.p_gain * prop + self.d_gain * velocity

        dist_next_wpnt = np.linalg.norm(location - next_wpnt)
        if (
            task_not_solved
            and (dist_next_wpnt < self.solve_thresh)
            and (vel_norm < self.vel_thresh)
        ):
            self._waypoint_idx += 1
            if self._waypoint_idx == len(self._waypoints) - 1:
                assert (
                    np.linalg.norm(self._waypoints[self._waypoint_idx] - self._target)
                    <= self.solve_thresh
                )

        self._waypoint_prev_loc = location
        action = np.clip(action, -1.0, 1.0)
        return action, (not task_not_solved)

    def gridify_state(self, state):
        return (int(round(state[0])), int(round(state[1])))

    def _new_target(self, start, target):
        # print('Computing waypoints from %s to %s' % (start, target))
        start = self.gridify_state(start)
        start_idx = self.env.gs.xy_to_idx(start)
        target = self.gridify_state(target)
        target_idx = self.env.gs.xy_to_idx(target)
        self._waypoint_idx = 0

        self.env.gs[target] = grid_spec.REWARD
        q_values = q_iteration.q_iteration(env=self.env, num_itrs=50, discount=0.99)
        # compute waypoints by performing a rollout in the grid
        max_ts = 100
        s = start_idx
        waypoints = []
        for i in range(max_ts):
            a = np.argmax(q_values[s])
            new_s, reward = self.env.step_stateless(s, a)

            waypoint = self.env.gs.idx_to_xy(new_s)
            if new_s != target_idx:
                waypoint = waypoint - np.random.uniform(size=(2,)) * 0.2
            waypoints.append(waypoint)
            s = new_s
            if new_s == target_idx:
                break
        self.env.gs[target] = grid_spec.EMPTY
        self._waypoints = waypoints
        self._waypoint_prev_loc = start
        self._target = target


if __name__ == "__main__":
    print(q_iteration.__file__)
    TEST_MAZE = "######\\" + "#OOOO#\\" + "#O##O#\\" + "#OOOO#\\" + "######"
    controller = WaypointController(TEST_MAZE)
    start = np.array((1, 1), dtype=np.float32)
    target = np.array((4, 3), dtype=np.float32)
    act, done = controller.get_action(start, target)
    print("wpt:", controller._waypoints)
    print(act, done)
    import pdb

    pdb.set_trace()
    pass
