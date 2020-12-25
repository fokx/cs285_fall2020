# part 1
### sanity check
We recommend usingLunarLander-v3to check the correctness of your code before running longer experimentswithMsPacman-v0.

To determine if your implementation of Q-learning is correct, you should run it with the default hyperparam-eters on theMs.  Pac-Mangame for 1 million steps using the command below.  Our reference solution gets areturn of 1500 in this timeframe.  On Colab, this will take roughly 3 GPU hours.  If it takes much longer thanthat, there may be a bug in your implementation.

To accelerate debugging, you may also test onLunarLander-v3, which trains your agent to play Lunar Lander,a 1979 arcade game (also made by Atari) that has been implemented in OpenAI Gym.  Our reference solutionwith the default hyperparameters achieves around 150 reward after 350k timesteps, but there is considerablevariation between runs and without the double-Q trick the average return often decreases after reaching 150.



### Question 1:  basic Q-learning performance.  (DQN)
Include a learning curve plot showing the per-formance of your implementation onMs.  Pac-Man.  The x-axis should correspond to number of time steps(consider using scientific notation) and the y-axis should show the average per-epoch reward as well as thebest mean reward so far.  These quantities are already computed and printed in the starter code.  They arealso logged to thedatafolder, and can be visualized using Tensorboard as in previous assignments.  Be sure tolabel the y-axis, since we need to verify that your implementation achieves similar reward as ours.  You shouldnot need to modify the default hyperparameters in order to obtain good performance, but if you modify anyof the parameters, list them in the caption of the figure.  The final results should use the following experimentname:

```shell
    /git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py  --env_name MsPacman-v0 --exp_name q1_ms_pacman
    
    /git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py  --env_name LunarLander-v3 --exp_name q1_lunar_lander
```


##  double Q-learning (DDQN)
Use the double estimator to improve the accuracy of yourlearned Q values.  This amounts to using the online Q network (instead of the target Q network) to select thebest action when computing target values.  Compare the performance of DDQN to vanilla DQN. Since there isconsiderable variance between runs, you must run at least three random seeds for both DQN and DDQN. Youmay uuseLunarLander-v3for this question.  The final results should use the following experiment names:

```shell
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1

/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2

/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3

/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --seed 1

/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --seed 2

/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --seed 3
```



 In your report, make a single graph thataverages the performance across three runs for both DQN and double DQN. Seescripts/readresults.pyfor an example of how to read the evaluation returns from Tensorboard logs.
 
## q3 Question 3:  experimenting with hyperparameters.Now 
let’s analyze the sensitivity of Q-learning tohyperparameters.   Choose  one  hyperparameter  of  your  choice  and  run  at  least  three  other  settings  of  thishyperparameter, in addition to the one used in Question 1, and plot all four values on the same graph.  Yourchoice what you experiment with, but you should explain why you chose this hyperparameter in the caption.Examples include:  learning rates, neural network architecture, exploration schedule or exploration rule (e.g.you may implement an alternative to -greedy), etc.  Discuss the effect of this hyperparameter on performancein the caption.  You should find a hyperparameter that makes a nontrivial difference on performance.  Note:you might consider performing a hyperparameter sweep for getting good results in Question 1, in which case it’sfine to just include the results of this sweep for Question 3 as well, while plotting only the best hyperparametersetting in Question 1.  The final results should use the following experiment name:

```shell
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam1
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam2
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam3

```

You can replaceLunarLander-v3withPongNoFrameskip-v4orMsPacman-v0if you would like to test on adifferent environment.

# part 2
## Question 4:  Sanity check with Cartpole
```shell
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1

```

In the example above, we alternate between performing one target update and one gradient update step for the critic.  As you will see, this probably doesn’t work, and you need to increase both the number of targetupdates and number of gradient updates.  Compare the results for the following settings and report whichworked best.  Do this by plotting all the runs on a single plot and writing your takeaway in the caption
```shell
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_100_1 -ntu 100 -ngsptu 1
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_1_100 -ntu 1 -ngsptu 100
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_10_10 -ntu 10 -ngsptu 10

```

At the end, the best setting from above should match the policy gradient results from Cartpole in hw2 (200).


## Question 5: Run actor-critic with more difficult tasks
Use the best setting from the previous questionto run InvertedPendulum and HalfCheetah:
10_10, 1_100 maybe better
```shell
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q5_10_10 -ntu 10 -ngsptu 10
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q5_1_100 -ntu 1 -ngsptu 100

/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_10_10 -ntu 10 -ngsptu 10
/git/py/env/rl0c/bin/python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_1_100 -ntu 1 -ngsptu 100
```

Your results should roughly match those of policy gradient.  After 150 iterations,  your HalfCheetah returnshould  be  around  150.   After  100  iterations,  your  InvertedPendulum  return  should  be  around  1000.   Yourdeliverables for this section are plots with the eval returns for both enviornments.
As a debugging tip,  the  returns  should start going up immediately.  For example,  after 20 iterations,  yourHalfCheetah return should be above -40 and your InvertedPendulum return should near or above 100. However,there is some variance between runs, so the 150-iteration (for HalfCheetah) and 100-iteration (for Inverted-Pendulum) results are the numbers we use to grade.