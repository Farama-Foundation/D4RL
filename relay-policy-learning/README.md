# Relay Policy Learning Environments

This is a set of environments and associated data for use with MuJoCo in a kitchen simulator.
The code instantiates a kitchen environment and parses associated demonstrations. 

## Getting Started (User)

1. Clone the repository
```
$ git clone https://adept.googlesource.com/adept_envs
```

2. Use the environments in your code (After including in the PYTHONPATH)
```
#!/usr/bin/env python3

import adept_envs
import gym

env = gym.make('dclaw3mx_track-v0')
```

This is not an officially supported Google product