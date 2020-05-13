"""Script for generating the datasets for kitchen environments."""
import d4rl.kitchen
import glob
import gym
import h5py
import numpy as np
import os
import pickle

np.set_printoptions(precision=2, suppress=True)

SAVE_DIRECTORY = '~/.offline_rl/datasets'
DEMOS_DIRECTORY = '~/relay-policy-learning/kitchen_demos_multitask'
DEMOS_SUBDIR_PATTERN = '*'
ENVIRONMENTS = ['kitchen_microwave_kettle_light_slider-v0',
                'kitchen_microwave_kettle_bottomburner_light-v0']
# Uncomment lines below for "mini_kitchen_microwave_kettle_light_slider-v0'".
DEMOS_SUBDIR_PATTERN = '*microwave_kettle_switch_slide'
ENVIRONMENTS = ['mini_kitchen_microwave_kettle_light_slider-v0']

OBS_ELEMENT_INDICES = [
    [11, 12],  # Bottom burners.
    [15, 16],  # Top burners.
    [17, 18],  # Light switch.
    [19],  # Slide.
    [20, 21],  # Hinge.
    [22],  # Microwave.
    [23, 24, 25, 26, 27, 28, 29],  # Kettle.
]
FLAT_OBS_ELEMENT_INDICES = sum(OBS_ELEMENT_INDICES, [])

def _relabel_obs_with_goal(obs_array, goal):
    obs_array[..., 30:] = goal
    return obs_array


def _obs_array_to_obs_dict(obs_array, goal=None):
    obs_dict = {
        'qp': obs_array[:9],
        'obj_qp': obs_array[9:30],
        'goal': goal,
    }
    if obs_dict['goal'] is None:
        obs_dict['goal'] = obs_array[30:]
    return obs_dict


def main():
    pattern = os.path.join(DEMOS_DIRECTORY, DEMOS_SUBDIR_PATTERN)
    demo_subdirs = sorted(glob.glob(pattern))
    print('Found %d demo subdirs.' % len(demo_subdirs))
    all_demos = {}
    for demo_subdir in demo_subdirs:
        demo_files = glob.glob(os.path.join(demo_subdir, '*.pkl'))
        print('Found %d demos in %s.' % (len(demo_files), demo_subdir))
        demos = []
        for demo_file in demo_files:
            with open(demo_file, 'rb') as f:
                demo = pickle.load(f)
            demos.append(demo)
        all_demos[demo_subdir] = demos

        # For debugging...
        all_observations = [demo['observations'] for demo in demos]
        first_elements = [obs[0, FLAT_OBS_ELEMENT_INDICES]
                          for obs in all_observations]
        last_elements = [obs[-1, FLAT_OBS_ELEMENT_INDICES]
                         for obs in all_observations]
        # End for debugging.

    for env_name in ENVIRONMENTS:
        env = gym.make(env_name).unwrapped
        env.REMOVE_TASKS_WHEN_COMPLETE = False  # This enables a Markovian reward.
        all_obs = []
        all_actions = []
        all_rewards = []
        all_terminals = []
        all_infos = []
        print('Relabelling data for %s.' % env_name)
        for demo_subdir, demos in all_demos.items():
            print('On demo from %s.' % demo_subdir)
            demos_obs = []
            demos_actions = []
            demos_rewards = []
            demos_terminals = []
            demos_infos = []
            for idx, demo in enumerate(demos):
                env_goal = env._get_task_goal()
                rewards = []
                relabelled_obs = _relabel_obs_with_goal(demo['observations'], env_goal)
                for obs in relabelled_obs:
                    reward_dict, score = env._get_reward_n_score(
                        _obs_array_to_obs_dict(obs))

                    rewards.append(reward_dict['r_total'])
                terminate_at = len(rewards)
                rewards = rewards[:terminate_at]
                demos_obs.append(relabelled_obs[:terminate_at])
                demos_actions.append(demo['actions'][:terminate_at])
                demos_rewards.append(np.array(rewards))
                demos_terminals.append(np.arange(len(rewards)) >= len(rewards) - 1)
                demos_infos.append([idx] * len(rewards))

            all_obs.append(np.concatenate(demos_obs))
            all_actions.append(np.concatenate(demos_actions))
            all_rewards.append(np.concatenate(demos_rewards))
            all_terminals.append(np.concatenate(demos_terminals))
            all_infos.append(np.concatenate(demos_infos))

            episode_rewards = [np.sum(rewards) for rewards in demos_rewards]
            last_rewards = [rewards[-1] for rewards in demos_rewards]
            print('Avg episode rewards %f.' % np.mean(episode_rewards))
            print('Avg last step rewards %f.' % np.mean(last_rewards))

        dataset_obs = np.concatenate(all_obs).astype('float32')
        dataset_actions = np.concatenate(all_actions).astype('float32')
        dataset_rewards = np.concatenate(all_rewards).astype('float32')
        dataset_terminals = np.concatenate(all_terminals).astype('float32')
        dataset_infos = np.concatenate(all_infos)
        dataset_size = len(dataset_obs)
        assert dataset_size == len(dataset_actions)
        assert dataset_size == len(dataset_rewards)
        assert dataset_size == len(dataset_terminals)
        assert dataset_size == len(dataset_infos)

        dataset = {
            'observations': dataset_obs,
            'actions': dataset_actions,
            'rewards': dataset_rewards,
            'terminals': dataset_terminals,
            'infos': dataset_infos,
            }

        print('Generated dataset with %d total steps.' % dataset_size)
        save_filename = os.path.join(SAVE_DIRECTORY, '%s.hdf5' % env_name)
        print('Saving dataset to %s.' % save_filename)
        h5_dataset = h5py.File(save_filename, 'w')
        for key in dataset:
            h5_dataset.create_dataset(key, data=dataset[key], compression='gzip')
        print('Done.')


if __name__ == '__main__':
    main()
