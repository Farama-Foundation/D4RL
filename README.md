# offline_rl
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


offline_rl is an open-source benchmark for offline reinforcement learning. It provides standardized environments and datasets for training and benchmarking algorithms.

## Setup

offline_rl can be installed by cloning the repository as follows:
```
git clone https://github.com/justinjfu/offline_rl.git
cd offline_rl
pip install -e .
```

Or, alternatively:
```
pip install git+https://github.com/justinjfu/offline_rl@master#egg=offline-rl
```

The control environments require MuJoCo as a dependency. You may need to obtain a [license](https://www.roboti.us/license.html) and follow the setup instructions for mujoco_py. This mostly involves copying the key to your MuJoCo installation folder.

## Using offline_rl

offline_rl uses the [OpenAI Gym](https://github.com/openai/gym) API. Tasks are created via the `gym.make` function.

Each task is associated with a fixed offline dataset, which can be obtained with the `get_dataset` method. This method returns an hdf5 dataset with `observations`, `actions`, `rewards`, `terminals`, and `infos` as keys. 

```python
import gym
import offline_rl # Import required to register environments

# Create the environment
env = gym.make('maze2d-umaze-v0')

# offline_rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation array of observations
```

Datasets are automatically downloaded to the `~/.offline_rl/datasets` directory. If you would like to change the location of this directory, you can set the `$OFFLINE_RL_DATASET_DIR` environment variable to the directory of your choosing, or pass in the dataset filepath directly into the `get_dataset` method.

## Acknowledgements

We would like to thank the authors of [hand_dapg](https://github.com/aravindr93/hand_dapg) and [gym-minigrid](https://github.com/maximecb/gym-minigrid) for providing several environments used in this repository.

