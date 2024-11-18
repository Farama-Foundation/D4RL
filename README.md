## Important Notice

### All of online environments libraries in D4RL have been moved [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [MiniGrid](https://github.com/Farama-Foundation/MiniGrid) and [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics), and all offline datasets in DR4L have been moved to [Minari](https://github.com/Farama-Foundation/Minari). These new versions include large bug fixes, new versions of Python, and are where all new development will continue. Please upgrade these libraries as soon as you're able to do so. If you'd like to read more about the story behind this switch, please check out [this blog post](https://farama.org/Announcing-Minari).

<p align="center">
    <img src="https://raw.githubusercontent.com/jjshoots/D4RL/master/d4rl-text.png" width="500px"/>
</p>

D4RL is an open-source benchmark for offline reinforcement learning. It provides standardized environments and datasets for training and benchmarking algorithms. A supplementary [whitepaper](https://arxiv.org/abs/2004.07219) and [website](https://sites.google.com/view/d4rl/home) are also available.

The current maintenance plan for this library is:

1) Pull the majority of the environments out of D4RL, fix the long standing bugs, and have them depend on the new MuJoCo bindings. The majority of the environments housed in D4RL were already maintained projects in Farama, and all the ones that aren't will be going into [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics), a standard library for housing many different Robotics environments. There are some envrionments that we don't plan to maintain, noteably the PyBullet ones (MuJoCo is not maintained and open source and PyBullet is now longer maintained) and Flow (it was never really used and the original author's don't view it as especially valuable).
2) Recreate all the datasets in D4RL given the revised versions of environments, and host them in a standard offline RL dataset repostiry we're working on called [Minari](https://github.com/Farama-Foundation/Minari).


## Setup

D4RL can be installed by cloning the repository as follows:
```
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```

Or, alternatively:
```
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

The control environments require MuJoCo as a dependency. You may need to obtain a [license](https://www.roboti.us/license.html) and follow the setup instructions for mujoco_py. This mostly involves copying the key to your MuJoCo installation folder.

The Flow and CARLA tasks also require additional installation steps:
- Instructions for installing CARLA can be found [here](https://github.com/Farama-Foundation/d4rl/wiki/CARLA-Setup)
- Instructions for installing Flow can be found [here](https://flow.readthedocs.io/en/latest/flow_setup.html). Make sure to install using the SUMO simulator, and add the flow repository to your PYTHONPATH once finished.

## Using d4rl

d4rl uses the [OpenAI Gym](https://github.com/openai/gym) API. Tasks are created via the `gym.make` function. A full list of all tasks is [available here](https://github.com/Farama-Foundation/d4rl/wiki/Tasks).

Each task is associated with a fixed offline dataset, which can be obtained with the `env.get_dataset()` method. This method returns a dictionary with:
- `observations`: An N by observation dimensional array of observations.
- `actions`: An N by action dimensional array of actions.
- `rewards`: An N dimensional array of rewards.
- `terminals`: An N dimensional array of episode termination flags. This is true when episodes end due to termination conditions such as falling over.
- `timeouts`: An N dimensional array of termination flags. This is true when episodes end due to reaching the maximum episode length.
- `infos`: Contains optional task-specific debugging information.

You can also load data using `d4rl.qlearning_dataset(env)`, which formats the data for use by typical Q-learning algorithms by adding a `next_observations` key.

```python
import gym
import d4rl # Import required to register environments, you may need to also import the submodule

# Create the environment
env = gym.make('maze2d-umaze-v1')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)
```

Datasets are automatically downloaded to the `~/.d4rl/datasets` directory when `get_dataset()` is called. If you would like to change the location of this directory, you can set the `$D4RL_DATASET_DIR` environment variable to the directory of your choosing, or pass in the dataset filepath directly into the `get_dataset` method.

### Normalizing Scores
You can use the `env.get_normalized_score(returns)` function to compute a normalized score for an episode, where `returns` is the undiscounted total sum of rewards accumulated during an episode.

The individual min and max reference scores are stored in `d4rl/infos.py` for reference.

## Algorithm Implementations

We have aggregated implementations of various offline RL algorithms in a [separate repository](https://github.com/Farama-Foundation/d4rl_evaluations).

## Off-Policy Evaluations

D4RL currently has limited support for off-policy evaluation methods, on a select few locomotion tasks. We provide trained reference policies and a set of performance metrics. Additional details can be found in the [wiki](https://github.com/Farama-Foundation/d4rl/wiki/Off-Policy-Evaluation).

## Recent Updates

### 2-12-2020
- Added new Gym-MuJoCo datasets (labeled v2) which fixed Hopper's performance and the qpos/qvel fields.
- Added additional wiki documentation on [generating datasets](https://github.com/Farama-Foundation/d4rl/wiki/Dataset-Reproducibility-Guide).


## Acknowledgements

D4RL builds on top of several excellent domains and environments built by various researchers. We would like to thank the authors of:
- [hand_dapg](https://github.com/aravindr93/hand_dapg)
- [gym-minigrid](https://github.com/maximecb/gym-minigrid)
- [carla](https://github.com/carla-simulator/carla)
- [flow](https://github.com/flow-project/flow)
- [adept_envs](https://github.com/google-research/relay-policy-learning)

## Citation

Please use the following bibtex for citations:

```
@misc{fu2020d4rl,
    title={D4RL: Datasets for Deep Data-Driven Reinforcement Learning},
    author={Justin Fu and Aviral Kumar and Ofir Nachum and George Tucker and Sergey Levine},
    year={2020},
    eprint={2004.07219},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Licenses

Unless otherwise noted, all datasets are licensed under the [Creative Commons Attribution 4.0 License (CC BY)](https://creativecommons.org/licenses/by/4.0/), and code is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html).


