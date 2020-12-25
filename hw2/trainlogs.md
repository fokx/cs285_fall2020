
##BAD 
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-12-18 --port 7000
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-03-47 --port 7002
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q2_b5000_r0.01_InvertedPendulum-v2_20-10-2020_14-00-03 --port 7003
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-26-54 --port 7004
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-26-33 --port 7010
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-12-03 --port 7012
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q3_b40000_r0.005_LunarLanderContinuous-v2_20-10-2020_14-02-38 --port 7016
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-49-51 --port 7017

`
`   
## OK:
### q3(continous)
      The policy performance may fluctuate around 1000; thisis fine.:
    /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q2_b5000_r0.01_InvertedPendulum-v2_21-10-2020_16-22-43
    /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q2_b5000_r0.01_InvertedPendulum-v2_21-10-2020_17-51-17

    
     You should expect to achieve an average return of around180 by the end of training:
    /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q3_b40000_r0.005_LunarLanderContinuous-v2_21-10-2020_15-02-24
    /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q3_b40000_r0.005_LunarLanderContinuous-v2_21-10-2020_17-51-22 # time: 8804

    /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q3_b40000_r0.005_LunarLanderContinuous-v2_09-11-2020_21-11-10 # time: 4104    

### q1(discrete)
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q1_sb_rtg_dsa_CartPole-v0_20-10-2020_13-59-47 --port 7001
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q1_sb_rtg_na_CartPole-v0_20-10-2020_14-01-14 --port 7005
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q1_lb_rtg_dsa_CartPole-v0_20-10-2020_14-07-28 --port 7007
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q1_lb_rtg_na_CartPole-v0_20-10-2020_14-14-13 --port 7011
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q1_lb_no_rtg_dsa_CartPole-v0_20-10-2020_14-02-45 --port 7013
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q1_sb_no_rtg_dsa_CartPole-v0_20-10-2020_13-52-03 --port 7015
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q1_sb_no_rtg_dsa_CartPole-v0_20-10-2020_13-58-46 --port 7018

##Maybe need more timesteps to learn?:
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-12-26 --port 7014

##Mysterious:
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_15-10-46 --port 7006
    tensorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-12-30 --port 7009

## Experiment4 HalfCheetah
### different hyperparameters(batch size, learning rate) comparison
best one: b50000_lr0.02: reaches 150 at timestep 70
    
    tensorsorboard --logdir /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_18-23-54 --port 7100   # reaches 130, and fall down to 0
    tensorsorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_18-09-00  --port 7101   # slow, only reaches 126 
    tensorsorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_18-08-50 --port 7102   # 60
    tensorsorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_19-14-47  --port 7103    # BEST
    tensorsorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_18-49-20  --port 7104   # only reaches 130
    tensorsorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_18-08-11  --port 7105   # -20
    tensorsorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_18-37-17  --port 7106   # 20
    tensorsorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_18-08-39  --port 7107   # 70
    tensorsorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_19-26-45  --port 7108' # too slow
    tensorsorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_21-10-2020_18-08-39/ --port 7109  # 60
 
### RUNS(comparison effects of reward to go, nn_baseline):
 reward to 
     python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 --exp_name q4_b50000_r0.02
      /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q4_b50000_r0.02_HalfCheetah-v2_24-10-2020_09-19-27
      noisy, less than 50

     python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --exp_name q4_b50000_r0.02_rtg
       /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q4_b50000_r0.02_rtg_HalfCheetah-v2_24-10-2020_09-19-31
            reaches 150 @ step90
     
     python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 --nn_baseline --exp_name q4_b50000_r0.02_nnbaseline
       /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q4_b50000_r0.02_nnbaseline_HalfCheetah-v2_24-10-2020_09-19-34
         noisy, less than 40
     
     python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name q4_b50000_r0.02_rtg_nnbaseline
          /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q4_b50000_r0.02_rtg_nnbaseline_HalfCheetah-v2_24-10-2020_09-19-38
         reaches 150 @ step70