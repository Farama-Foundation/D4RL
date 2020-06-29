"""Open merge example.
Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
on-ramp merge to a single lane open highway network.
"""
from flow.envs import MergePOEnv
from flow.networks import MergeNetwork
from copy import deepcopy
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, RLController

def gen_env(render='drgb'):
    # time horizon of a single rollout
    HORIZON = 750
    # inflow rate at the highway
    FLOW_RATE = 2000
    # percent of autonomous vehicles
    RL_PENETRATION = 0.1
    # num_rl term (see ADDITIONAL_ENV_PARAMs)
    NUM_RL = 5

    # We consider a highway network with an upstream merging lane producing
    # shockwaves
    additional_net_params = deepcopy(ADDITIONAL_NET_PARAMS)
    additional_net_params["merge_lanes"] = 1
    additional_net_params["highway_lanes"] = 1
    additional_net_params["pre_merge_length"] = 500

    # RL vehicles constitute 5% of the total number of vehicles
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=9,
        ),
        num_vehicles=5)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=9,
        ),
        num_vehicles=0)

    # Vehicles are introduced from both sides of merge, with RL vehicles entering
    # from the highway portion as well
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
        depart_lane="free",
        depart_speed=10)
    inflow.add(
        veh_type="rl",
        edge="inflow_highway",
        vehs_per_hour=RL_PENETRATION * FLOW_RATE,
        depart_lane="free",
        depart_speed=10)
    inflow.add(
        veh_type="human",
        edge="inflow_merge",
        vehs_per_hour=100,
        depart_lane="free",
        depart_speed=7.5)

    flow_params = dict(
        # name of the experiment
        exp_tag="merge_0",

        # name of the flow environment the experiment is running on
        env_name=MergePOEnv,

        # name of the network class the experiment is running on
        network=MergeNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            restart_instance=True,
            sim_step=0.5,
            render=render,
            save_render=True
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            sims_per_step=2,
            warmup_steps=0,
            additional_params={
                "max_accel": 1.5,
                "max_decel": 1.5,
                "target_velocity": 20,
                "num_rl": NUM_RL,
            },
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
        initial=InitialConfig(),
    )
    return flow_params
