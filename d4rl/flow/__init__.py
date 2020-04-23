import gym
from d4rl import offline_env
from gym.envs.registration import register

from copy import deepcopy

import flow
import flow.envs
from flow.networks.ring import RingNetwork
from flow.core.params import NetParams, VehicleParams, EnvParams
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers import RLController
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.envs.ring.accel import AccelEnv
from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
from flow.envs import WaveAttenuationPOEnv


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


def example_flow_env():
    name = "training_example"
    network_name = RingNetwork

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

    sim_params = SumoParams(sim_step=0.1, render=False)
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

    env_name = WaveAttenuationPOEnv

    flow_params = dict(
        # name of the experiment
        exp_tag=name,
        # name of the flow environment the experiment is running on
        env_name=env_name,
        # name of the network class the experiment uses
        network=network_name,
        # simulator that is used by the experiment
        simulator='traci',
        # simulation-related parameters
        sim=sim_params,
        # environment related parameters (see flow.core.params.EnvParams)
        env=env_params,
        # network-related parameters (see flow.core.params.NetParams and
        # the network's documentation or ADDITIONAL_NET_PARAMS component)
        net=net_params,
        # vehicles to be placed in the network at the start of a rollout 
        # (see flow.core.vehicles.Vehicles)
        veh=vehicles,
        # (optional) parameters affecting the positioning of vehicles upon 
        # initialization/reset (see flow.core.params.InitialConfig)
        initial=initial_config
    )
    return flow_params


register(
    id='flow-env-v0',
    entry_point='d4rl.flow:flow_register',
    kwargs={
        'flow_params': example_flow_env(),
        'dataset_url': None,
        'ref_min_score': None,
        'ref_max_score': None
    }
)
