# A baseline town agent.
from agents.navigation.agent import Agent, AgentState
import numpy as np
from agents.navigation.local_planner import LocalPlanner

class RoamingAgent(Agent):
    """
    RoamingAgent implements a basic agent that navigates scenes making random
    choices when facing an intersection.

    This agent respects traffic lights and other vehicles.

    NOTE: need to re-create after each env reset
    """

    def __init__(self, env):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        vehicle = env.vehicle
        follow_traffic_lights = env.follow_traffic_lights
        super(RoamingAgent, self).__init__(vehicle)
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._local_planner = LocalPlannerModified(self._vehicle)
        self._follow_traffic_lights = follow_traffic_lights

    def compute_action(self):
        action, traffic_light = self.run_step()
        throttle = action.throttle
        brake = action.brake
        steer = action.steer
        #print('tbsl:', throttle, brake, steer, traffic_light)
        if brake == 0.0:
            return np.array([throttle, steer])
        else:
            return np.array([-brake, steer])

    def run_step(self):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step()

        throttle = control.throttle
        brake = control.brake
        steer = control.steer
        #print('tbsl:', throttle, brake, steer, traffic_light)
        if brake == 0.0:
            return np.array([throttle, steer])
        else:
            return np.array([-brake, steer])


class LocalPlannerModified(LocalPlanner):

    def __del__(self):
        pass  # otherwise it deletes our vehicle object

    def run_step(self):
        return super().run_step(debug=False)  # otherwise by default shows waypoints, that interfere with our camera


class DummyTownAgent(Agent):
    """
    A simple agent for the town driving task.

    If the car is currently facing on a path towards the goal, drive forward.
    If the car would start drivign away, apply maximum brakes.
    """

    def __init__(self, env):
        """
        :param vehicle: actor to apply to local planner logic onto
        """
        self.env = env
        super(DummyTownAgent, self).__init__(self.env.vehicle)
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._local_planner = LocalPlannerModified(self._vehicle)

    def compute_action(self):

        hazard_detected = False
        # retrieve relevant elements for safe navigation, i.e.: traffic lights and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")
        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True



        rotation = self.env.vehicle.get_transform().rotation
        forward_vector = rotation.get_forward_vector()
        origin = self.env.vehicle.get_location()
        destination = self.env.target_location
        node_list = self.env.route_planner._path_search(origin=origin, destination=destination)
        origin_xy = np.array([origin.x, origin.y])
        forward_xy = np.array([forward_vector.x, forward_vector.y])
        first_node_xy = self.env.route_planner._graph.nodes[node_list[0]]['vertex']
        first_node_xy = np.array([first_node_xy[0], first_node_xy[1]])
        target_direction_vector = first_node_xy - origin_xy
        target_unit_vector = np.array(target_direction_vector) / np.linalg.norm(target_direction_vector)
        vel_s = np.dot(forward_xy, target_unit_vector)
        if vel_s < 0:
            hazard_detected = True


        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step()
        throttle = control.throttle
        brake = control.brake
        steer = control.steer
        #print('tbsl:', throttle, brake, steer, traffic_light)
        if brake == 0.0:
            return np.array([throttle, steer])
        else:
            return np.array([-brake, steer])
