from gym.envs.registration import register
from importlib_metadata import entry_points
from d4rl.gym_bullet import gym_envs
from d4rl import infos


scenario_configs = {
    'Full-Easy-Stochastic-v0': {
        'map_name': '11_vs_11_easy_stochastic',
        'n_right_agents': 11,
        'n_left_agents': 11
    },
    'Half-v0': {
        'map_name': '5_vs_5',
        'n_right_agents': 5,
        'n_left_agents': 5
    },
    'Single-v0': {
        'map_name': '1_vs_1_easy',
        'n_right_agents': 1,
        'n_left_agents': 1
    }
}


for scenario, configs in scenario_configs.items():
    register(
        id=scenario,
        entry_point='d4rl.grfootball.grf_envs:get_env',
        max_episode_steps=3000,
        kwargs={
            'scenario_config': {
                **configs
            }
        }
    )

    # for dataset in ['random', 'medium', 'expert', 'medium-expert', 'medium-replay']:
    #     env_name = 'grf-%s-%s-v0' % (scenario, dataset)
    #     register(
    #         id=env_name,
    #         entry_point='d4rl.grfootball.grf_envs:get_env',
    #         max_episode_steps=3000,
    #         kwargs={
    #             'scenario_config': {
    #                 **configs
    #             },
    #             'ref_min_score': infos.REF_MIN_SCORE[env_name],
    #             'ref_max_score': infos.REF_MAX_SCORE[env_name],
    #             'dataset_url': infos.DATASET_URLS[env_name]
    #         }
    #     )