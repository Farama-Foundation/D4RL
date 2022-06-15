from gym.envs.registration import register


scenario_configs = {
    'Full-Easy-Stochastic-v0': {
        'map_name': '11_vs_11_easy_stochastic',
        'n_right_players': 11,
        'n_left_players': 11
    },
    'Half-v0': {
        'map_name': '5_vs_5',
        'n_right_players': 5,
        'n_left_players': 5
    },
    'Single-v0': {
        'map_name': '1_vs_1_easy',
        'n_right_players': 1,
        'n_left_players': 1
    }
}


register(
    id='GRFootball-v0',
    entry_point='d4rl.grfootball.grf_envs:get_env',
    # kwargs={
    #     'scenario_config': {
    #         **configs
    #     }
    # }
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