## speed comparison of parallel data collection
When A has finishied(4104s), B has just come to Iteration 58's collecting data for training phase!
```shell script
A. Use parallel trajectories collection and train using only CPU:
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005 --no_gpu

B. Collecting trajectories single with thread and training using GPU 
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005

```


## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions. If you completed this installation for homework 1, you do not need to repeat it.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2020/blob/master/hw2/cs285/scripts/run_hw2.ipynb)
\begin{align}\pi^*(s) = \arg\!\max_a \ Q^*(s, a)\end{align}
## Complete the code

The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with "TODO: get this from hw1".
```shell script
/home/hawk/py/env/py38.cd11.0/bin/python3.8 /home/hawk/dl/cs285/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa
########################
logging outputs to  /home/hawk/dl/cs285/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q1_sb_no_rtg_dsa_CartPole-v0_17-09-2020_21-44-19
########################
GPU not detected. Defaulting to CPU.

```
- [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](cs285/infrastructure/utils.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)

You will then need to complete the following new files for homework 2. The relevant sections are marked with "TODO".
- [agents/pg_agent.py](cs285/agents/pg_agent.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)

You will also want to look through [scripts/run_hw2.py](cs285/scripts/run_hw2.py) (if running locally) or [scripts/run_hw2.ipynb](cs285/scripts/run_hw1.2pynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

You will be running your policy gradients implementation in four experiments total, investigating the effects of design decisions like reward-to-go estimators, neural network baselines for variance reduction, and advantage normalization. See the [assignment PDF](cs285_hw2.pdf) for more details.

## Plotting your results

We have provided a snippet that may be used for reading your Tensorboard eventfiles in [scripts/read_results.py](cs285/scripts/read_results.py). Reading these eventfiles and plotting them with [matplotlib](https://matplotlib.org/) or [seaborn](https://seaborn.pydata.org/) will produce the cleanest results for your submission. For debugging purposes, we recommend visualizing the Tensorboard logs using `tensorboard --logdir data`.
