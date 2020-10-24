
## Question 4 remarks:
        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
             for j in range(self.num_grad_steps_per_target_update):
                # update target for ntu(num_target_updates) times in total
             # do gradient steps for ngsptu * ntu times in total:
             # (num_grad_steps_per_target_update * num_target_updates)
so, all 4 settings do 100 gradient steps except the first one(1x1) for that batch
* ntu=100, ngsptu=1: update target immediately after one gradient step 
(time: 99s)
* ntu=1, ngsptu=100: update target only once 
(far more time: 235)
* ntu=10, ngsptu=10: update target every 10 gradient steps
(time: 109s)

I guess the [ntu=100, ngsptu=1] option is the best choice 

Results:

### cmds used:
    python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_100_1 -ntu 100 -ngsptu 1

    python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_1_100 -ntu 1 -ngsptu 100


## Question 5
cmd example:
python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_cheetah_1_100 -ntu 1 -ngsptu 100

### invertedPendulum: reaches 800
    tensorboard --logdir  /git/py.code/hw3/homework_fall2020/hw3/cs285/data/hw3_q5_100_1_InvertedPendulum-v2_21-10-2020_20-45-39/  --port 7100 # 4m52s
    tensorboard --logdir  /git/py.code/hw3/homework_fall2020/hw3/cs285/data/hw3_q5_10_10_InvertedPendulum-v2_21-10-2020_20-44-54/  --port 7101 # 4m39s
    tensorboard --logdir  /git/py.code/hw3/homework_fall2020/hw3/cs285/data/hw3_q5_1_100_InvertedPendulum-v2_21-10-2020_21-31-09/  --port 7102 # 7m24s

### halfCheetah: reaches 120
the shapes of the 4 curves of AverageReturn almost have no difference. the time used @step 149 is not comparable as they are run at different time

    tensorboard --logdir  /git/py.code/hw3/homework_fall2020/hw3/cs285/data/hw3_q5_cheetah_100_1_HalfCheetah-v2_21-10-2020_20-47-37/  --port 7100 # 42m12
    tensorboard --logdir  /git/py.code/hw3/homework_fall2020/hw3/cs285/data/hw3_q5_cheetah_10_19_HalfCheetah-v2_21-10-2020_20-47-46/  --port 7101 # 42m23
    tensorboard --logdir  /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q5_cheetah_1_100_HalfCheetah-v2_24-10-2020_09-42-41  --port 7102 # 49m40s
    tensorboard --logdir  /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q5_cheetah_10_10_HalfCheetah-v2_24-10-2020_09-26-51  --port 7103 # 51m50s
  
python /git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_cheetah_1_100 -ntu 1 -ngsptu 100
