# exp1
```shell
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py  --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa # dsa=dont_standardize_advantages
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py  --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa 
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py  --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na # na=normalize advantage
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py  --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py  --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py  --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na

```
n:  Number of iterations.•-b:  Batch size (number of state-action pairs sampled while acting according to the current policy ateach iteration).•-dsa:  Flag:  if present,  setsstandardize_advantagesto False.  Otherwise,  by default,  standardizesadvantages to have a mean of zero and standard deviation of one.•-rtg:  Flag:  if present, setsreward_to_go=True.  Otherwise,reward_to_go=Falseby default.•--exp_name:  Name for experiment, which goes into the name for the data logging directory.

Create two graphs:–In the first graph, compare the learning curves (average return at each iteration) for the experimentsprefixed withq1_sb_.  (The small batch experiments.)–In the second graph, compare the learning curves for the experiments prefixed withq1_lb_.  (Thelarge batch experiments.)•Answer the following questions briefly:–Which value estimator has better performance without advantage-standardization:  the trajectory-centric one, or the one using reward-to-go?–Did advantage standardization help?–Did the batch size make an impact?

The best configuration of CartPole in both the large and small batch cases should converge to a maximumscore of 200

# exp2
```shell

/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.0025 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.0025_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.005_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.01_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.02_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.03 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.03_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.0025 -rtg --nn_baseline --exp_name q4_search_b20000_lr0.0025_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b20000_lr0.005_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b20000_lr0.01_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b20000_lr0.02_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.03 -rtg --nn_baseline --exp_name q4_search_b20000_lr0.03_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.0025 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.0025_rtg_nnbaseline 
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.005_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.01_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.02_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.03 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.03_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.0025 -rtg --nn_baseline --exp_name q4_search_b40000_lr0.0025_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b40000_lr0.005_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b40000_lr0.01_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b40000_lr0.02_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.03 -rtg --nn_baseline --exp_name q4_search_b40000_lr0.03_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.0025 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.0025_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.005_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.01_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.02_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.03 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.03_rtg_nnbaseline

```

Without nn_baseline:
```shell
# start
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.0025 -rtg  --exp_name q2_b10000_lr0.0025_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.005 -rtg  --exp_name q2_b10000_lr0.005_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.01 -rtg  --exp_name q2_b10000_lr0.01_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.02 -rtg  --exp_name q2_b10000_lr0.02_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.03 -rtg  --exp_name q2_b10000_lr0.03_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.0025 -rtg  --exp_name q2_b20000_lr0.0025_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.005 -rtg  --exp_name q2_b20000_lr0.005_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.01 -rtg  --exp_name q2_b20000_lr0.01_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.02 -rtg  --exp_name q2_b20000_lr0.02_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 20000 -lr 0.03 -rtg  --exp_name q2_b20000_lr0.03_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.0025 -rtg  --exp_name q2_b30000_lr0.0025_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.005 -rtg  --exp_name q2_b30000_lr0.005_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.01 -rtg  --exp_name q2_b30000_lr0.01_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.02 -rtg  --exp_name q2_b30000_lr0.02_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 30000 -lr 0.03 -rtg  --exp_name q2_b30000_lr0.03_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.0025 -rtg  --exp_name q2_b40000_lr0.0025_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg  --exp_name q2_b40000_lr0.005_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.01 -rtg  --exp_name q2_b40000_lr0.01_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.02 -rtg  --exp_name q2_b40000_lr0.02_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 40000 -lr 0.03 -rtg  --exp_name q2_b40000_lr0.03_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.0025 -rtg  --exp_name q2_b50000_lr0.0025_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.005 -rtg  --exp_name q2_b50000_lr0.005_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.01 -rtg  --exp_name q2_b50000_lr0.01_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.02 -rtg  --exp_name q2_b50000_lr0.02_rtg
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.03 -rtg  --exp_name q2_b50000_lr0.03_rtg


```
Given theb*andr*you found, provide a learning curve where the policy gets to optimum (maximumscore  of   1000)  in  less  than  100  iterations.   (This  may  be  for  a  single  random  seed,  or  averaged  overmultiple.)

```shell

```
# exp3
```shell
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005
```
Plot a learning curve for the above command.  You should expect to achieve an average return of around180 by the end of training
# exp4
ou will be using your policy gradient implementation to learn a controllerfor theHalfCheetah-v2benchmark environment with an episode length of 150. This is shorter than the defaultepisode length (1000), which speeds up training significantly.  Search over batch sizesb∈[10000,30000,50000]and learning ratesr∈[0.005,0.01,0.02] to replace<b>and<r>below.python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \--discount 0.95 -n 100 -l 2 -s 32 -b <b> -lr <r> -rtg --nn_baseline \--exp_name q4_search_b<b>_lr<r>_rtg_nnbaseline
```shell
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.005_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.01_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.02_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.005_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.01_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.02_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.005_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.01_rtg_nnbaseline
/git/py/env/rl0b/bin/python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.02_rtg_nnbaseline
```
