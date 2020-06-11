#!/usr/bin/env python

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
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import is_within_distance_ahead #, is_within_distance, compute_distance
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle

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


class CustomGlobalRoutePlanner(GlobalRoutePlanner):
    def __init__(self, dao):
        super(CustomGlobalRoutePlanner, self).__init__(dao=dao)

    """
    def compute_distance(self, origin, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)
        distance = 0.0
        for idx in range(len(node_list) - 1):
            distance += (super(CustomGlobalRoutePlanner, self)._distance_heuristic(node_list[idx], node_list[idx+1]))
        # print ('Distance: ', distance)
        return distance
    """

    def compute_direction_velocities(self, origin, velocity, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)

        origin_xy = np.array([origin.x, origin.y])
        velocity_xy = np.array([velocity.x, velocity.y])

        first_node_xy = self._graph.nodes[node_list[1]]['vertex']
        first_node_xy = np.array([first_node_xy[0], first_node_xy[1]])
        target_direction_vector = first_node_xy - origin_xy
        target_unit_vector = np.array(target_direction_vector) / np.linalg.norm(target_direction_vector)

        vel_s = np.dot(velocity_xy, target_unit_vector)

        unit_velocity = velocity_xy / (np.linalg.norm(velocity_xy) + 1e-8)
        angle = np.arccos(np.clip(np.dot(unit_velocity, target_unit_vector), -1.0, 1.0))
        vel_perp = np.linalg.norm(velocity_xy) * np.sin(angle)
        return vel_s, vel_perp

    def compute_distance(self, origin, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)
        #print('Node list:', node_list)
        first_node_xy = self._graph.nodes[node_list[0]]['vertex']
        #print('Diff:', origin, first_node_xy)

        #distance = 0.0
        distances = []
        distances.append(np.linalg.norm(np.array([origin.x, origin.y, 0.0]) - np.array(first_node_xy)))

        for idx in range(len(node_list) - 1):
            distances.append(super(CustomGlobalRoutePlanner, self)._distance_heuristic(node_list[idx], node_list[idx+1]))
        #print('Distances:', distances)
        #import pdb; pdb.set_trace()
        return np.sum(distances)

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


class CarlaEnv(object):

    def __init__(self, args):
        self.render_display = False
        self.record_display = False
        self.record_vision = True
        self.record_dir = None #'/nfs/kun1/users/aviralkumar/carla_data/'
        self.vision_size = args.vision_size
        self.vision_fov = args.vision_fov
        self.changing_weather_speed = float(args.weather)
        self.frame_skip = args.frame_skip
        self.max_episode_steps = args.steps
        self.multiagent = args.multiagent
        self.start_lane = args.lane
        self.follow_traffic_lights = args.lights
        if self.record_display:
            assert self.render_display

        self.actor_list = []

        if self.render_display:
            pygame.init()
            self.render_display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()
        self.map = self.world.get_map()

        ## Define the route planner
        self.route_planner_dao = GlobalRoutePlannerDAO(self.map, sampling_resolution=0.1) 
        self.route_planner = CustomGlobalRoutePlanner(self.route_planner_dao)

        # tests specific to map 4:
        if self.start_lane and self.map.name != "Town04":
            raise NotImplementedError

        # remove old vehicles and sensors (in case they survived)
        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter("*vehicle*"):
            print("Warning: removing old vehicle")
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print("Warning: removing old sensor")
            sensor.destroy()

        self.vehicle = None
        self.vehicles_list = []  # their ids
        self.reset_vehicle()  # creates self.vehicle
        self.actor_list.append(self.vehicle)

        blueprint_library = self.world.get_blueprint_library()

        if self.render_display:
            self.camera_display = self.world.spawn_actor(
                blueprint_library.find('sensor.camera.rgb'),
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
            self.actor_list.append(self.camera_display)

        bp = blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.vision_size))
        bp.set_attribute('image_size_y', str(self.vision_size))
        bp.set_attribute('fov', str(self.vision_fov))
        location = carla.Location(x=1.6, z=1.7)
        self.camera_vision = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
        self.actor_list.append(self.camera_vision)

        if self.record_display or self.record_vision:
            if self.record_dir is None:
                self.record_dir = "carla-{}-{}x{}-fov{}".format(
                    self.map.name.lower(), self.vision_size, self.vision_size, self.vision_fov)
                if self.frame_skip > 1:
                    self.record_dir += '-{}'.format(self.frame_skip)
                if self.changing_weather_speed > 0.0:
                    self.record_dir += '-weather'
                if self.multiagent:
                    self.record_dir += '-mutiagent'
                if self.follow_traffic_lights:
                    self.record_dir += '-lights'
                self.record_dir += '-{}k'.format(self.max_episode_steps // 1000)

                now = datetime.datetime.now()
                self.record_dir += now.strftime("-%Y-%m-%d-%H-%M-%S")
            if not os.path.exists(self.record_dir):
                os.mkdir(self.record_dir)

        if self.render_display:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_display, self.camera_vision, fps=20)
        else:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_vision, fps=20)

        # weather
        self.weather = Weather(self.world, self.changing_weather_speed)

        # dummy variables, to match deep mind control's APIs
        low = -1.0
        high = 1.0
        self.action_space = DotMap()
        self.action_space.low.min = lambda: low
        self.action_space.high.max = lambda: high
        self.action_space.shape = [2]
        self.observation_space = DotMap()
        self.observation_space.shape = (3, self.vision_size, self.vision_size)
        self.observation_space.dtype = np.dtype(np.uint8)
        self.reward_range = None
        self.metadata = None
        self.action_space.sample = lambda: np.random.uniform(low=low, high=high, size=self.action_space.shape[0]).astype(np.float32)

        # roaming carla agent
        self.agent = None
        self.world.tick()
        self.reset_init()  # creates self.agent

        ## Initialize the route planner
        self.route_planner.setup()

        ## Collision detection
        self._proximity_threshold = 10.0
        self._traffic_light_threshold = 5.0
        self.actor_list = self.world.get_actors()
        for idx in range(len(self.actor_list)):
            print (idx, self.actor_list[idx])
        # import ipdb; ipdb.set_trace()
        self.vehicle_list = self.actor_list.filter("*vehicle*")
        self.lights_list = self.actor_list.filter("*traffic_light*")
        self.object_list = self.actor_list.filter("*traffic.*")

        ## Initialize the route planner
        self.route_planner.setup()

        ## The map is deterministic so for reward relabelling, we can
        ## instantiate the environment object and then query the distance function
        ## in the env, which directly uses this map_graph, and we need not save it.
        self._map_graph = self.route_planner._graph

        ## This is a dummy for the target location, we can make this an input
        ## to the env in RL code.
        self.target_location = carla.Location(x=-13.473097, y=134.311234, z=-0.010433)

        ## Now reset the env once
        self.reset()
        
        
    def reset_init(self):
        self.reset_vehicle()
        self.world.tick()
        self.reset_other_vehicles()
        self.world.tick()
        self.agent = RoamingAgent(self.vehicle, follow_traffic_lights=self.follow_traffic_lights)
        self.count = 0
        self.ts = int(time.time())

    def reset(self):
        # get obs:
        obs, _, _, _ = self.step()
        return obs

    def reset_vehicle(self):

        if self.map.name == "Town04":
            start_lane = -1
            start_x = 5.0
            vehicle_init_transform = carla.Transform(carla.Location(x=start_x, y=0, z=0.1), carla.Rotation(yaw=-90))
        else:
            init_transforms = self.world.get_map().get_spawn_points()
            vehicle_init_transform = random.choice(init_transforms)

        # TODO(aviral): start lane not defined for town, also for the town, we may not want to have
        # the lane following reward, so it should be okay.

        if self.vehicle is None:  # then create the ego vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find('vehicle.audi.a2')
            self.vehicle = self.world.spawn_actor(vehicle_blueprint, vehicle_init_transform)

        self.vehicle.set_transform(vehicle_init_transform)
        self.vehicle.set_velocity(carla.Vector3D())
        self.vehicle.set_angular_velocity(carla.Vector3D())

    def reset_other_vehicles(self):
        if not self.multiagent:
            return

        # clear out old vehicles
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []

        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.set_synchronous_mode(True)
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        num_vehicles = 20
        if self.map.name == "Town04":
            road_id = 47
            road_length = 117.
            init_transforms = []
            for _ in range(num_vehicles):
                lane_id = random.choice([-1, -2, -3, -4])
                vehicle_s = np.random.uniform(road_length)  # length of road 47
                init_transforms.append(self.map.get_waypoint_xodr(road_id, lane_id, vehicle_s).transform)
        else:
            init_transforms = self.world.get_map().get_spawn_points()
            init_transforms = np.random.choice(init_transforms, num_vehicles)

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for transform in init_transforms:
            transform.location.z += 0.1  # otherwise can collide with the road it starts on
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True)))

        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)

        traffic_manager.global_percentage_speed_difference(30.0)

    def compute_action(self):
        return self.agent.run_step()

    def step(self, action=None, traffic_light_color=""):
        rewards = []
        for _ in range(self.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(action, traffic_light_color)
            rewards.append(reward)
            if done:
                break
        return next_obs, np.mean(rewards), done, info
    
    def _is_vehicle_hazard(self, vehicle, vehicle_list):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self.map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_vehicle.get_transform(),
                                        vehicle.get_transform(),
                                        self._proximity_threshold/10.0):
                return (True, -1.0, target_vehicle)

        return (False, 0.0,  None)

    def _is_object_hazard(self, vehicle, object_list):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for target_vehicle in object_list:
            # do not account for the ego vehicle
            if target_vehicle.id == vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self.map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_vehicle.get_transform(),
                                        vehicle.get_transform(),
                                        self._proximity_threshold/40.0):
                return (True, -1.0, target_vehicle)

        return (False, 0.0,  None)
    
    def _is_light_red(self, vehicle):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.
        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for traffic_light in self.lights_list:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self.map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(object_waypoint.transform,
                                        vehicle.get_transform(),
                                        self._traffic_light_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, -0.1, traffic_light)

        return (False, 0.0, None)
    
    def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    def _get_collision_reward(self, vehicle):
        vehicle_hazard, reward, vehicle_id = self._is_vehicle_hazard(vehicle, self.vehicle_list)
        return vehicle_hazard, reward
    
    def _get_traffic_light_reward(self, vehicle):
        traffic_light_hazard, reward, traffic_light_id = self._is_light_red(vehicle)
        return traffic_light_hazard, 0.0
    
    def _get_object_collided_reward(self, vehicle):
        object_hazard, reward, object_id = self._is_object_hazard(vehicle, self.object_list)
        return object_hazard, reward
    
    def goal_reaching_reward(self, vehicle):
        # Now we will write goal_reaching_rewards
        vehicle_location = vehicle.get_location()
        target_location = self.target_location

        # This is the distance computation
        """
        dist = self.route_planner.compute_distance(vehicle_location, target_location)

        base_reward = -1.0 * dist
        collided_done, collision_reward = self._get_collision_reward(vehicle)
        traffic_light_done, traffic_light_reward = self._get_traffic_light_reward(vehicle)
        object_collided_done, object_collided_reward = self._get_object_collided_reward(vehicle)
        total_reward = base_reward + 100 * collision_reward + 100 * traffic_light_reward + 100.0 * object_collided_reward
        """

        vehicle_velocity = vehicle.get_velocity()
        dist = self.route_planner.compute_distance(vehicle_location, target_location)
        vel_forward, vel_perp = self.route_planner.compute_direction_velocities(vehicle_location, vehicle_velocity, target_location)
        #print('[GoalReachReward] VehLoc: %s Target: %s Dist: %s VelF:%s' % (str(vehicle_location), str(target_location), str(dist), str(vel_forward)))
        #base_reward = -1.0 * (dist / 100.0) + 5.0
        base_reward = vel_forward
        collided_done, collision_reward = self._get_collision_reward(vehicle)
        traffic_light_done, traffic_light_reward = self._get_traffic_light_reward(vehicle)
        object_collided_done, object_collided_reward = self._get_object_collided_reward(vehicle)
        total_reward = base_reward + 100 * collision_reward # + 100 * traffic_light_reward + 100.0 * object_collided_reward

        reward_dict = dict()
        reward_dict['collision'] = collision_reward
        reward_dict['traffic_light'] = traffic_light_reward
        reward_dict['object_collision'] = object_collided_reward
        reward_dict['base_reward'] = base_reward
        reward_dict['vel_forward'] = vel_forward
        reward_dict['vel_perp'] = vel_perp
        done_dict = dict()
        done_dict['collided_done'] = collided_done
        done_dict['traffic_light_done'] = traffic_light_done
        done_dict['object_collided_done'] = object_collided_done
        return total_reward, reward_dict, done_dict

    def _simulator_step(self, action, traffic_light_color):

        if self.render_display:
            if should_quit():
                return
            self.clock.tick()

        if action is None:
            throttle, steer, brake = 0., 0., 0.
        else:
            throttle, steer, brake = action.throttle, action.steer, action.brake
            # throttle = clamp(throttle, minimum=0.005, maximum=0.995) + np.random.uniform(low=-0.003, high=0.003)
            # steer = clamp(steer, minimum=-0.995, maximum=0.995) + np.random.uniform(low=-0.003, high=0.003)
            # brake = clamp(brake, minimum=0.005, maximum=0.995) + np.random.uniform(low=-0.003, high=0.003)

            vehicle_control = carla.VehicleControl(
                throttle=throttle,  # [0,1]
                steer=steer,  # [-1,1]
                brake=brake,  # [0,1]
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )
            self.vehicle.apply_control(vehicle_control)

        # Advance the simulation and wait for the data.
        if self.render_display:
            snapshot, display_image, vision_image = self.sync_mode.tick(timeout=2.0)
        else:
            snapshot, vision_image = self.sync_mode.tick(timeout=2.0)

        # Weather evolves
        self.weather.tick()

        # Draw the display.
        if self.render_display:
            draw_image(self.render_display, display_image)
            self.render_display.blit(self.font.render('Frame %d' % self.count, True, (255, 255, 255)), (8, 10))
            self.render_display.blit(self.font.render('Control: %5.2f thottle, %5.2f steer, %5.2f brake' % (throttle, steer, brake), True, (255, 255, 255)), (8, 28))
            self.render_display.blit(self.font.render('Traffic light: ' + traffic_light_color, True, (255, 255, 255)), (8, 46))
            self.render_display.blit(self.font.render(str(self.weather), True, (255, 255, 255)), (8, 64))
            pygame.display.flip()

        # Format rl image
        bgra = np.array(vision_image.raw_data).reshape(self.vision_size, self.vision_size, 4)  # BGRA format
        bgr = bgra[:, :, :3]  # BGR format (84 x 84 x 3)
        rgb = np.flip(bgr, axis=2)  # RGB format (84 x 84 x 3)

        reward, reward_dict, done_dict = self.goal_reaching_reward(self.vehicle)

        if self.render_display and self.record_display:
            image_name = os.path.join(self.record_dir, "display%08d.jpg" % self.count)
            pygame.image.save(self.render_display, image_name)
            # # Can animate with:
            # ffmpeg -r 20 -pattern_type glob -i 'display*.jpg' carla.mp4
        if self.record_vision:
            image_name = os.path.join(self.record_dir, "vision_%d_%08d.png" % (self.ts, self.count))
            im = Image.fromarray(rgb)
            # add any eta data you like into the image before we save it:
            metadata = PngInfo()
            # control
            metadata.add_text("control_throttle", str(throttle))
            metadata.add_text("control_steer", str(steer))
            metadata.add_text("control_brake", str(brake))
            metadata.add_text("control_repeat", str(self.frame_skip))
            # acceleration
            acceleration = self.vehicle.get_acceleration()
            metadata.add_text("acceleration_x", str(acceleration.x))
            metadata.add_text("acceleration_y", str(acceleration.y))
            metadata.add_text("acceleration_z", str(acceleration.z))
            # angular velocity
            angular_velocity = self.vehicle.get_angular_velocity()
            metadata.add_text("angular_velocity_x", str(angular_velocity.x))
            metadata.add_text("angular_velocity_y", str(angular_velocity.y))
            metadata.add_text("angular_velocity_z", str(angular_velocity.z))
            # location
            location = self.vehicle.get_location()
            print('Location:', location)
            metadata.add_text("location_x", str(location.x))
            metadata.add_text("location_y", str(location.y))
            metadata.add_text("location_z", str(location.z))
            # rotation
            rotation = self.vehicle.get_transform().rotation
            metadata.add_text("rotation_pitch", str(rotation.pitch))
            metadata.add_text("rotation_yaw", str(rotation.yaw))
            metadata.add_text("rotation_roll", str(rotation.roll))
            forward_vector = rotation.get_forward_vector()
            metadata.add_text("forward_vector_x", str(forward_vector.x))
            metadata.add_text("forward_vector_y", str(forward_vector.y))
            metadata.add_text("forward_vector_z", str(forward_vector.z))
            # velocity
            velocity = self.vehicle.get_velocity()
            metadata.add_text("velocity_x", str(velocity.x))
            metadata.add_text("velocity_y", str(velocity.y))
            metadata.add_text("velocity_z", str(velocity.z))
            # weather
            metadata.add_text("weather_cloudiness ", str(self.weather.weather.cloudiness))
            metadata.add_text("weather_precipitation", str(self.weather.weather.precipitation))
            metadata.add_text("weather_precipitation_deposits", str(self.weather.weather.precipitation_deposits))
            metadata.add_text("weather_wind_intensity", str(self.weather.weather.wind_intensity))
            metadata.add_text("weather_fog_density", str(self.weather.weather.fog_density))
            metadata.add_text("weather_wetness", str(self.weather.weather.wetness))
            metadata.add_text("weather_sun_azimuth_angle", str(self.weather.weather.sun_azimuth_angle))
            # settings
            metadata.add_text("settings_map", self.map.name)
            metadata.add_text("settings_vision_size", str(self.vision_size))
            metadata.add_text("settings_vision_fov", str(self.vision_fov))
            metadata.add_text("settings_changing_weather_speed", str(self.changing_weather_speed))
            metadata.add_text("settings_multiagent", str(self.multiagent))
            # traffic lights
            metadata.add_text("traffic_lights_color", "UNLABELED")
            metadata.add_text("reward", str(reward))

            ## Add in reward dict
            for key in reward_dict:
                metadata.add_text("reward_" + str(key), str(reward_dict[key]))
            
            for key in done_dict:
                metadata.add_text("done_" + str(key), str(done_dict[key]))

            ## Save the target location as well
            metadata.add_text('target_location_x', str(self.target_location.x))
            metadata.add_text('target_location_y', str(self.target_location.y))
            metadata.add_text('target_location_z', str(self.target_location.z))

            im.save(image_name, "PNG", pnginfo=metadata)

            # # To read these images later, you can run something like this:
            # from PIL.PngImagePlugin import PngImageFile
            # im = PngImageFile("vision00001234.png")
            # throttle = float(im.text['throttle'])  # range [0, 1]
            # steer = float(im.text['steer'])  # range [-1, 1]
            # brake = float(im.text['brake'])  # range [0, 1]
            # lights = im.text['lights']  # traffic lights color, [NONE, JUNCTION, RED, YELLOW, GREEN]
        self.count += 1

        next_obs = rgb  # 84 x 84 x 3
        # # To inspect images, run:
        # import pdb; pdb.set_trace()
        # import matplotlib.pyplot as plt
        # plt.imshow(next_obs)
        # plt.show()

        done = False #self.count >= self.max_episode_steps
        if done:
            print("Episode success: I've reached the episode horizon ({}).".format(self.max_episode_steps))
        # print ('reward: ', reward)
        info = reward_dict
        info.update(done_dict)
        done = False
        for key in done_dict:
            done = (done or done_dict[key])
        return next_obs, reward, done, info

    def finish(self):
        print('destroying actors.')
        for actor in self.actor_list:
            actor.destroy()
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        time.sleep(0.5)
        pygame.quit()
        print('done.')


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
    """

    def __init__(self, vehicle, follow_traffic_lights=True):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(RoamingAgent, self).__init__(vehicle)
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._local_planner = LocalPlannerModified(self._vehicle)
        self._follow_traffic_lights = follow_traffic_lights

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

    curr_steps = 0
    try:
        done = False
        while not done:
            curr_steps += 1 
            action, traffic_light_color = env.compute_action()
            next_obs, reward, done, info = env.step(action, traffic_light_color)
            print ('Reward: ', reward, 'Done: ', done, 'Location: ', env.vehicle.get_location())
            if done:
                # env.reset_init()
                # env.reset()
                done = False
            
            if curr_steps % 5000 == 4999:
                env.reset_init()
                env.reset()
    finally:
        env.finish()
