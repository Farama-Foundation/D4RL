import gym
import os
from d4rl import offline_env
from gym.envs.registration import register

from copy import deepcopy

import flow
import flow.envs
from flow.networks.ring import RingNetwork
from flow.networks.minicity import MiniCityNetwork
from flow.networks.bay_bridge_toll import BayBridgeTollNetwork
from flow.core.params import NetParams, VehicleParams, EnvParams, InFlows
from flow.core.params import SumoLaneChangeParams, SumoCarFollowingParams
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter 
from flow.controllers import SimCarFollowingController, BayBridgeRouter, SimLaneChangeController
from flow.controllers import RLController
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.envs.ring.accel import AccelEnv
from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
from flow.envs import WaveAttenuationPOEnv
from flow.envs import BayBridgeEnv, TrafficLightGridPOEnv
from flow.envs import BottleneckDesiredVelocityEnv
from flow.networks import BottleneckNetwork

import d4rl.flow.traffic_light_grid as traffic_light_grid

def flow_register(flow_params, render=None, **kwargs):
    exp_tag = flow_params["exp_tag"]
    env_params = flow_params['env']
    net_params = flow_params['net']
    env_class = flow_params['env_name']
    initial_config = flow_params.get('initial', InitialConfig())
    traffic_lights = flow_params.get("tls", TrafficLightParams())
    sim_params = deepcopy(flow_params['sim'])
    vehicles = deepcopy(flow_params['veh'])

    sim_params.render = render or sim_params.render

    if isinstance(flow_params["network"], str):
        print("""Passing of strings for network will be deprecated.
        Please pass the Network instance instead.""")
        module = __import__("flow.networks", fromlist=[flow_params["network"]])
        network_class = getattr(module, flow_params["network"])
    else:
        network_class = flow_params["network"]

    network = network_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights,
    )

    flow_env = env_class(
        env_params= env_params,
        sim_params= sim_params,
        network= network,
        simulator= flow_params['simulator']
    )

    env = offline_env.OfflineEnvWrapper(flow_env,
        **kwargs
    )
    return env


def ring_env(render='drgb'):
    name = "ring"
    network_name = RingNetwork
    env_name = WaveAttenuationPOEnv

    net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
    initial_config = InitialConfig(spacing="uniform", perturbation=1)

    vehicles = VehicleParams()
    vehicles.add("human",
                 acceleration_controller=(IDMController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=21)
    vehicles.add(veh_id="rl",
                 acceleration_controller=(RLController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=1)

    sim_params = SumoParams(sim_step=0.5, render=render, save_render=True)
    HORIZON=100
    env_params = EnvParams(
        # length of one rollout
        horizon=HORIZON,
        additional_params={
            # maximum acceleration of autonomous vehicles
            "max_accel": 1,
            # maximum deceleration of autonomous vehicles
            "max_decel": 1,
            # bounds on the ranges of ring road lengths the autonomous vehicle 
            # is trained on
            "ring_length": [220, 270],
        },
    )


    flow_params = dict(
        exp_tag=name,
        env_name=env_name,
        network=network_name,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config
    )
    return flow_params


def bottleneck(render='drgb', SCALING=1):
    HORIZON=1000
    DISABLE_TB = True
    DISABLE_RAMP_METER = True
    AV_FRAC = 0.10

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="all_checks",
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,
        ),
        num_vehicles=1 * SCALING)
    vehicles.add(
        veh_id="followerstopper",
        acceleration_controller=(RLController, {}),
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=9,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,
        ),
    num_vehicles=1 * SCALING)

    controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                           ("4", 2, True), ("5", 1, False)]
    num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
    additional_env_params = {
        "target_velocity": 40,
        "disable_tb": True,
        "disable_ramp_metering": True,
        "controlled_segments": controlled_segments,
        "symmetric": False,
        "observed_segments": num_observed_segments,
        "reset_inflow": False,
        "lane_change_duration": 5,
        "max_accel": 3,
        "max_decel": 3,
        "inflow_range": [1000, 2000]
    }

    # flow rate
    flow_rate = 2300 * SCALING

    # percentage of flow coming out of each lane
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehs_per_hour=flow_rate * (1 - AV_FRAC),
        departLane="random",
        departSpeed=10)
    inflow.add(
        veh_type="followerstopper",
        edge="1",
        vehs_per_hour=flow_rate * AV_FRAC,
        departLane="random",
        departSpeed=10)

    traffic_lights = TrafficLightParams()
    if not DISABLE_TB:
        traffic_lights.add(node_id="2")
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id="3")

    additional_net_params = {"scaling": SCALING, "speed_limit": 23}
    net_params = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    flow_params = dict(
        # name of the experiment
        exp_tag="bottleneck",

        # name of the flow environment the experiment is running on
        env_name=BottleneckDesiredVelocityEnv,

        # name of the network class the experiment is running on
        network=BottleneckNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.5,
            render=render,
            save_render=True,
            print_warnings=False,
            restart_instance=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            warmup_steps=40,
            sims_per_step=1,
            horizon=HORIZON,
            additional_params=additional_env_params,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing="uniform",
            min_gap=5,
            lanes_distribution=float("inf"),
            edges_distribution=["2", "3", "4", "5"],
        ),

        # traffic lights to be introduced to specific nodes (see
        # flow.core.params.TrafficLightParams)
        tls=traffic_lights,
    )

    return flow_params


register(
    id='flow-ring-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=500,
    kwargs={
        'flow_params': ring_env(render=False),
        'dataset_url': None,
        'ref_min_score': None,
        'ref_max_score': None
    }
)


register(
    id='flow-ring-render-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=500,
    kwargs={
        'flow_params': ring_env(render='drgb'),
        'dataset_url': None,
        'ref_min_score': None,
        'ref_max_score': None
    }
)


register(
    id='flow-trafficlight-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=200,
    kwargs={
        'flow_params': traffic_light_grid.gen_env(render=False),
        'dataset_url': None,
        'ref_min_score': None,
        'ref_max_score': None
    }
)


register(
    id='flow-trafficlight-render-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=200,
    kwargs={
        'flow_params': traffic_light_grid.gen_env(render='drgb'),
        'dataset_url': None,
        'ref_min_score': None,
        'ref_max_score': None
    }
)


register(
    id='flow-bottleneck-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=1000,
    kwargs={
        'flow_params': bottleneck(render=False),
        'dataset_url': None,
        'ref_min_score': None,
        'ref_max_score': None
    }
)


register(
    id='flow-bottleneck-render-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=1000,
    kwargs={
        'flow_params': bottleneck(render='drgb'),
        'dataset_url': None,
        'ref_min_score': None,
        'ref_max_score': None
    }
)
