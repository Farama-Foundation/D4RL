# Scripts for training BRAC

This directory provides some scripts to train a [BRAC](https://arxiv.org/abs/1911.11361) agent on the offline RL datasets provided in this library. 

## Setup

Download the BRAC code:
```
mkdir google_research
cd google_research/
svn export https://github.com/google-research/google-research/trunk/behavior_regularized_offline_rl
```
Be sure to add the resulting `google_research` directory to your PYTHONPATH.

## Running

First, train a behavior policy. See `run_bc.sh` for an example.

Once a behavior policy is trained, you may then train a behavior-regularized actor critic agent. See `run_primal.sh` for an example.
