# !/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Modified by Rowan McAllister on 20 April 2020

import argparse
import datetime
import glob
import os
import random
import sys
import time
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import math

from dotmap import DotMap

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within a certain distance from a reference object.
    A vehicle in front would be something around 0 deg, while one behind around 180 deg.
        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :param d_angle_th_up: upper thereshold for angle
        :param d_angle_th_low: low thereshold for angle (optional, default is 0)
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points
        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm



class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

        self.start()

    def start(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        min_alt, max_alt = [20, 90]
        self.altitude = 0.5 * (max_alt + min_alt) + 0.5 * (max_alt - min_alt) * math.cos(self._t)

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 60.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, world, changing_weather_speed):
        self.world = world
        self.reset()
        self.weather = world.get_weather()
        self.changing_weather_speed = changing_weather_speed
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

    def reset(self):
        weather_params = carla.WeatherParameters(sun_altitude_angle=90.)
        self.world.set_weather(weather_params)

    def tick(self):
        self._sun.tick(self.changing_weather_speed)
        self._storm.tick(self.changing_weather_speed)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude
        self.world.set_weather(self.weather)

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_size', type=int, default=84)
    parser.add_argument('--vision_fov', type=int, default=90)
    parser.add_argument('--weather', default=False, action='store_true')
    parser.add_argument('--frame_skip', type=int, default=1),
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--multiagent', default=False, action='store_true'),
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--lights', default=False, action='store_true')
    args = parser.parse_args()
    return args


class LocalPlannerModified(LocalPlanner):

    def __del__(self):
        pass  # otherwise it deletes our vehicle object

    def run_step(self):
        return super().run_step(debug=False)  # otherwise by default shows waypoints, that interfere with our camera


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
        traffic_light_color = self._is_light_red(lights_list)
        if traffic_light_color == 'RED' and self._follow_traffic_lights:
            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step()

        #print ('Action chosen: ', control)
        return control, traffic_light_color

    # override case class
    def _is_light_red_europe_style(self, lights_list):
        """
        This method is specialized to check European style traffic lights.
        Only suitable for Towns 03 -- 07.
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        traffic_light_color = "NONE"  # default, if no traffic lights are seen

        for traffic_light in lights_list:
            object_waypoint = self._map.get_waypoint(traffic_light.get_location())
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(traffic_light.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._proximity_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return "RED"
                elif traffic_light.state == carla.TrafficLightState.Yellow:
                    traffic_light_color = "YELLOW"
                elif traffic_light.state == carla.TrafficLightState.Green:
                    if traffic_light_color is not "YELLOW":  # (more severe)
                        traffic_light_color = "GREEN"
                else:
                    import pdb; pdb.set_trace()
                    # investigate https://carla.readthedocs.io/en/latest/python_api/#carlatrafficlightstate

        return traffic_light_color

    # override case class
    def _is_light_red_us_style(self, lights_list, debug=False):
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        traffic_light_color = "NONE"  # default, if no traffic lights are seen

        if ego_vehicle_waypoint.is_junction:
            # It is too late. Do not block the intersection! Keep going!
            return "JUNCTION"

        if self._local_planner.target_waypoint is not None:
            if self._local_planner.target_waypoint.is_junction:
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(loc,
                                                               ego_vehicle_location,
                                                               self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 60.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if debug:
                        print('=== Magnitude = {} | Angle = {} | ID = {}'.format(
                            sel_magnitude, min_angle, sel_traffic_light.id))

                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.TrafficLightState.Red:
                        return "RED"
                    elif self._last_traffic_light.state == carla.TrafficLightState.Yellow:
                        traffic_light_color = "YELLOW"
                    elif self._last_traffic_light.state == carla.TrafficLightState.Green:
                        if traffic_light_color is not "YELLOW":  # (more severe)
                            traffic_light_color = "GREEN"
                    else:
                        import pdb; pdb.set_trace()
                        # investigate https://carla.readthedocs.io/en/latest/python_api/#carlatrafficlightstate
                else:
                    self._last_traffic_light = None

        return traffic_light_color


if __name__ == '__main__':

    # example call:
    # ./PythonAPI/util/config.py --map Town01 --delta-seconds 0.05
    # python PythonAPI/carla/agents/navigation/data_collection_agent.py --vision_size 256 --vision_fov 90 --steps 10000 --weather --lights

    args = parse_args()
    env = CarlaEnv(args)

    try:
        done = False
        while not done:
            action, traffic_light_color = env.compute_action()
            next_obs, reward, done, info = env.step(action, traffic_light_color)
            print ('Reward: ', reward, 'Done: ', done, 'Location: ', env.vehicle.get_location())
            if done:
                # env.reset_init()
                # env.reset()
                done = False

    finally:
        env.finish()
