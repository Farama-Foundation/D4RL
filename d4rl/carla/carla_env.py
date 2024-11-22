import argparse
import datetime
import glob
import os
import random
import sys
import time
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import gym
from gym import Env
import gym.spaces as spaces

#from . import proxy_env
from d4rl.offline_env import OfflineEnv

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

# This is CARLA agent
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
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
    Euclidean distance between 3D po-0.427844-0.427844ints
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

    def compute_direction_velocities(self, origin, velocity, destination):
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)

        origin_xy = np.array([origin.x, origin.y])
        velocity_xy = np.array([velocity.x, velocity.y])
        first_node_xy = self._graph.nodes[node_list[0]]['vertex']
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
        first_node_xy = self._graph.nodes[node_list[1]]['vertex']
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

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

## Now the actual env
class CarlaEnv(object):
    """
    CARLA agent, we will wrap this in a proxy env to get a gym env
    """
    def __init__(self, render=False, carla_port=2000, record=False, record_dir=None, args=None, record_vision=False, reward_type='lane_follow', **kwargs):
        self.render_display = render
        self.record_display = record
        print('[CarlaEnv] record_vision:', record_vision)
        self.record_vision = record_vision
        self.record_dir = record_dir
        self.reward_type = reward_type
        self.vision_size = args['vision_size']
        self.vision_fov = args['vision_fov']
        self.changing_weather_speed = float(args['weather'])
        self.frame_skip = args['frame_skip']
        self.max_episode_steps = args['steps']  # DMC uses this
        self.multiagent = args['multiagent']
        self.start_lane = args['lane']
        self.follow_traffic_lights = args['lights']
        if self.record_display:
            assert self.render_display

        self.actor_list = []

        if self.render_display:
            pygame.init()
            self.render_display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()

        self.client = carla.Client('localhost', carla_port)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()
        self.map = self.world.get_map()

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
        
        self.action_space = spaces.Box(low=np.array((low, low)), high=np.array((high, high)))

        self.observation_space = DotMap()
        self.observation_space.shape = (3, self.vision_size, self.vision_size)
        self.observation_space.dtype = np.dtype(np.uint8)
        self.reward_range = None
        self.metadata = None
        # self.action_space.sample = lambda: np.random.uniform(low=low, high=high, size=self.action_space.shape[0]).astype(np.float32)

        self.horizon = self.max_episode_steps
        self.image_shape = (3, self.vision_size, self.vision_size)

        # roaming carla agent
        self.count = 0
        self.world.tick()
        self.reset_init()

        self._proximity_threshold = 10.0
        self._traffic_light_threshold = 5.0
        self.actor_list = self.world.get_actors()
        #for idx in range(len(self.actor_list)):
        #    print (idx, self.actor_list[idx])

        # import ipdb; ipdb.set_trace()
        self.vehicle_list = self.actor_list.filter("*vehicle*")
        self.lights_list = self.actor_list.filter("*traffic_light*")
        self.object_list = self.actor_list.filter("*traffic.*")

        # town nav
        self.route_planner_dao = GlobalRoutePlannerDAO(self.map, sampling_resolution=0.1) 
        self.route_planner = CustomGlobalRoutePlanner(self.route_planner_dao)
        self.route_planner.setup()
        self.target_location = carla.Location(x=-13.473097, y=134.311234, z=-0.010433)

        # roaming carla agent
        # self.agent = None
        # self.count = 0
        # self.world.tick()
        self.reset()  # creates self.agent

    
    def reset_init(self):
        self.reset_vehicle()
        self.world.tick()
        self.reset_other_vehicles()
        self.world.tick()

        #

        self.count = 0

    def reset(self):
        #self.reset_vehicle()
        #self.world.tick()
        #self.reset_other_vehicles()
        #self.world.tick()
        #self.count = 0
        # get obs:
        #for _ in range(5):
        #    self.world.tick()
            #obs, _, _, _ = self.step()

        obs, _, done, _ = self.step()

        # keep resetting until vehicle is not collided
        total_resets = 0
        while done:
            self.reset_vehicle()
            self.world.tick()
            obs, _, done, _ = self.step()
            total_resets += 1
            if total_resets > 10:
                break

        return obs
    
    def reset_vehicle(self):

        if self.map.name == "Town04":
            self.start_lane = -1 # np.random.choice([-1, -2, -3, -4])  # their positive values, not negative
            start_x = 5.
            vehicle_init_transform = carla.Transform(carla.Location(x=start_x, y=0, z=0.1), carla.Rotation(yaw=-90))
        else:
            init_transforms = self.world.get_map().get_spawn_points()
            vehicle_init_transform = random.choice(init_transforms)
            #print('MyInitTransform', vehicle_init_transform)
        

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
            #print('OtherInitTransforms:')
            #for transf in init_transforms:
            #    print(transf)

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
    
    def step(self, action=None, traffic_light_color=""):
        """
        rewards = []
        for _ in range(self.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(action, traffic_light_color)
            rewards.append(reward)
            if done:
                break
        return next_obs, np.mean(rewards), done, info
        """
        return self._simulator_step(action, traffic_light_color)
    
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

        # Check the lane ids
        loc = vehicle.get_location() 
        if loc is not None:
            w = self.map.get_waypoint(loc)
            if w is not None:
                current_lane_id = w.lane_id
                if current_lane_id not in [-1, 1]:
                    #print ('Lane: ', current_lane_id, self.start_lane)
                    vehicle_hazard = True
                    reward = -1.0
            else:
                vehicle_hazard = True
                reward = -1.0
        else:
            vehicle_hazard = True
            reward = -1.0

        #print ('vehicle: ', loc, current_lane_id, self.start_lane)
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
        vehicle_velocity = vehicle.get_velocity()

        target_location = self.target_location

        # This is the distance computation
        try:
            dist = self.route_planner.compute_distance(vehicle_location, target_location)
            vel_forward, vel_perp = self.route_planner.compute_direction_velocities(vehicle_location, vehicle_velocity, target_location)
        except TypeError:
            # Weird bug where the graph disappears
            vel_forward = 0
            vel_perp = 0
        
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
        done_dict = dict()
        done_dict['collided_done'] = collided_done
        done_dict['traffic_light_done'] = traffic_light_done
        done_dict['object_collided_done'] = object_collided_done
        return total_reward, reward_dict, done_dict

    def lane_follow_reward(self, vehicle):
        # assume on highway
        vehicle_location = vehicle.get_location()
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        vehicle_xy = np.array([vehicle_location.x, vehicle_location.y])
        vehicle_s = vehicle_waypoint.s
        vehicle_velocity = vehicle.get_velocity()  # Vector3D
        vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
        # print ('Velocity: ', vehicle_velocity_xy)
        speed = np.linalg.norm(vehicle_velocity_xy)
        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        assert road_id is not None
        goal_abs_lane_id = 1  # just for goal-following
        lane_id_sign = int(np.sign(vehicle_waypoint_closest_to_road.lane_id))
        assert lane_id_sign in [-1, 1]
        goal_lane_id = goal_abs_lane_id * lane_id_sign
        current_waypoint = self.map.get_waypoint(vehicle_location, project_to_road=False)
        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)

        # Check for valid goal waypoint
        if goal_waypoint is None:
            print ('goal waypoint is None...')
            # try to fix, bit of a hack, with CARLA waypoint discretizations
            carla_waypoint_discretization = 0.02  # meters
            goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s - carla_waypoint_discretization)
            if goal_waypoint is None:
                goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s + carla_waypoint_discretization)

        # set distance to 100 if the waypoint is off the road
        if goal_waypoint is None:
            print("Episode fail: goal waypoint is off the road! (frame %d)" % self.count)
            done, dist, vel_s = True, 100., 0.
        else:
            goal_location = goal_waypoint.transform.location
            goal_xy = np.array([goal_location.x, goal_location.y])
            # dist = np.linalg.norm(vehicle_xy - goal_xy)
            dists = []
            for abs_lane_id in [1, 2, 3, 4]:
                lane_id_ = abs_lane_id * lane_id_sign
                wp = self.map.get_waypoint_xodr(road_id, lane_id_, vehicle_s)
                if wp is not None:  # lane 4 might not exist where the highway has a turnoff
                    loc = wp.transform.location
                    xy = np.array([loc.x, loc.y])
                    dists.append(np.linalg.norm(vehicle_xy - xy))
            if dists:
                dist = min(dists)  # just try to get to the center of one of the lanes
            else:
                dist = 0.
            next_goal_waypoint = goal_waypoint.next(0.1)  # waypoints are ever 0.02 meters
            if len(next_goal_waypoint) != 1:
                print('warning: {} waypoints (not 1)'.format(len(next_goal_waypoint)))
            if len(next_goal_waypoint) == 0:
                print("Episode done: no more waypoints left. (frame %d)" % self.count)
                done, vel_s, vel_perp = True, 0., 0.
            else:
                location_ahead = next_goal_waypoint[0].transform.location
                highway_vector = np.array([location_ahead.x, location_ahead.y]) - goal_xy
                highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
                vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)

                unit_velocity = vehicle_velocity_xy / (np.linalg.norm(vehicle_velocity_xy) + 1e-8)
                angle = np.arccos(np.clip(np.dot(unit_velocity, highway_unit_vector), -1.0, 1.0))
                #vel_forward = np.linalg.norm(vehicle_velocity_xy) * np.cos(angle)
                vel_perp = np.linalg.norm(vehicle_velocity_xy) * np.sin(angle)
                #print('R:', np.clip(vel_s-5*vel_perp, -5.0, 5.0), 'vel_s:', vel_s, 'vel_perp:', vel_perp)
                #import pdb; pdb.set_trace()

                done = False

        # not algorithm's fault, but the simulator sometimes throws the car in the air wierdly
        # usually in initial few frames, which can be ignored
        """
        if vehicle_velocity.z > 1. and self.count < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_velocity.z, self.count))
            done = True
        if vehicle_location.z > 0.5 and self.count < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_location.z, self.count))
            done = True
        """

        ## Add rewards for collision and optionally traffic lights
        vehicle_location = vehicle.get_location()
        base_reward = np.clip(vel_s - 5*vel_perp, -5.0, 5.0)
        collided_done, collision_reward = self._get_collision_reward(vehicle)
        traffic_light_done, traffic_light_reward = self._get_traffic_light_reward(vehicle)
        object_collided_done, object_collided_reward = self._get_object_collided_reward(vehicle)
        total_reward = base_reward + 100 * collision_reward + 100 * traffic_light_reward + 100.0 * object_collided_reward
        reward_dict = dict()
        reward_dict['collision'] = collision_reward
        reward_dict['traffic_light'] = traffic_light_reward
        reward_dict['object_collision'] = object_collided_reward
        reward_dict['base_reward'] = base_reward
        reward_dict['base_reward_vel_s'] = vel_s
        reward_dict['base_reward_vel_perp'] = vel_perp
        done_dict = dict()
        done_dict['collided_done'] = collided_done
        done_dict['traffic_light_done'] = traffic_light_done
        done_dict['object_collided_done'] = object_collided_done
        done_dict['base_done'] = done
        return total_reward, reward_dict, done_dict
    
    def _simulator_step(self, action, traffic_light_color):
        
        if action is None:
            throttle, steer, brake = 0., 0., 0.
        else:
            steer = float(action[1])
            throttle_brake = float(action[0])

            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake

            vehicle_control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer), 
                brake=float(brake),
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
            self.render_display.blit(self.font.render('Frame %d' % self.count, True, (255, 255, 255)), (8, 10))
            self.render_display.blit(self.font.render('Control: %5.2f thottle, %5.2f steer, %5.2f brake' % (throttle, steer, brake), True, (255, 255, 255)), (8, 28))
            self.render_display.blit(self.font.render('Traffic light: ' + traffic_light_color, True, (255, 255, 255)), (8, 46))
            self.render_display.blit(self.font.render(str(self.weather), True, (255, 255, 255)), (8, 64))
            pygame.display.flip()

        # Format rl image
        bgra = np.array(vision_image.raw_data).reshape(self.vision_size, self.vision_size, 4)  # BGRA format
        bgr = bgra[:, :, :3]  # BGR format (84 x 84 x 3)
        rgb = np.flip(bgr, axis=2)  # RGB format (84 x 84 x 3)

        if self.render_display and self.record_display:
            image_name = os.path.join(self.record_dir, "display%08d.jpg" % self.count)
            pygame.image.save(self.render_display, image_name)
            # # Can animate with:
            # ffmpeg -r 20 -pattern_type glob -i 'display*.jpg' carla.mp4
        if self.record_vision:
            image_name = os.path.join(self.record_dir, "vision%08d.png" % self.count)
            print('savedimg:', image_name)
            im = Image.fromarray(rgb)

            # add any meta data you like into the image before we save it:
            metadata = PngInfo()
            metadata.add_text("throttle", str(throttle))
            metadata.add_text("steer", str(steer))
            metadata.add_text("brake", str(brake))
            metadata.add_text("lights", traffic_light_color)

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

        self.count += 1

        next_obs = rgb 
        
        done = False
        if done:
            print("Episode success: I've reached the episode horizon ({}).".format(self.max_episode_steps))

        if self.reward_type=='lane_follow':
            reward, reward_dict, done_dict = self.lane_follow_reward(self.vehicle)
        elif self.reward_type=='goal_reaching':
            reward, reward_dict, done_dict = self.goal_reaching_reward(self.vehicle)
        else:
            raise ValueError('unknown reward type:', self.reward_type)

        info = reward_dict
        info.update(done_dict)
        done = False
        for key in done_dict:
            done = (done or done_dict[key])
        #if done:
        #    print('done_dict:', done_dict, 'r:', reward)
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


class CarlaObsDictEnv(OfflineEnv):
    def __init__(self, carla_args=None, carla_port=2000, reward_type='lane_follow', render_images=False, **kwargs):
        self._wrapped_env = CarlaEnv(carla_port=carla_port, args=carla_args, reward_type=reward_type, record_vision=render_images)
        self._wrapped_env = CarlaEnv(carla_port=carla_port, args=carla_args, record_vision=render_images)
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

        obs_shape = self._wrapped_env.observation_space.shape  # 3x48x48
        obs_shape = np.roll(obs_shape, -1)
        self.observation_space = spaces.Dict({
            'image':spaces.Box(low=np.zeros(obs_shape, dtype=np.float32), high=np.ones(obs_shape, dtype=np.float32))
        })
        super(CarlaObsDictEnv, self).__init__(**kwargs)

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        self._wrapped_env.reset_init()
        obs = (self._wrapped_env.reset(**kwargs))
        obs_dict = dict()
        # Also normalize obs
        obs = np.transpose(obs, [1, 2, 0])
        obs_dict['image'] = (obs.astype(np.float32) / 255.0)
        return obs_dict

    def step(self, action):
        #print ('Action: ', action)
        next_obs, reward, done, info = self._wrapped_env.step(action)
        next_obs_dict = dict()
        next_obs = np.transpose(next_obs, [1, 2, 0])
        next_obs_dict['image'] = (next_obs.astype(np.float32) / 255.0)
        return next_obs_dict, reward, done, info

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self._wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class CarlaObsEnv(OfflineEnv):
    def __init__(self, carla_args=None, carla_port=2000, reward_type='lane_follow', render_images=False, **kwargs):
        self._wrapped_env = CarlaEnv(carla_port=carla_port, args=carla_args, reward_type=reward_type, record_vision=render_images)
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space
        obs_shape = self._wrapped_env.observation_space.shape
        obs_shape = np.roll(obs_shape, -1)
        self.observation_space = spaces.Dict({
            'image':spaces.Box(low=np.zeros(obs_shape, dtype=np.float32), high=np.ones(obs_shape, dtype=np.float32))
        })
        super(CarlaObsEnv, self).__init__(**kwargs)

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        self._wrapped_env.reset_init()
        obs = (self._wrapped_env.reset(**kwargs))
        obs = np.transpose(obs, [1, 2, 0])
        # Also normalize obs
        obs = (obs.astype(np.float32) / 255.0)
        return obs

    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        next_obs = np.transpose(next_obs, [1, 2, 0])
        next_obs = (next_obs.astype(np.float32) / 255.0)
        return next_obs, reward, done, info

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self._wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

if __name__ == '__main__':
    variant = dict()
    variant['vision_size'] = 48
    variant['vision_fov'] = 48
    variant['weather'] = False
    variant['frame_skip'] = 1
    variant['steps'] = 100000
    variant['multiagent'] = False
    variant['lane'] = 0
    variant['lights'] = False
    variant['record_dir'] = None

    env = CarlaEnv(args=variant)
    carla_gym_env = proxy_env.ProxyEnv(env)
