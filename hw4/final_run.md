# q1
```shell

/git/py/env/rl0d/bin/python /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch1x32 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1 --no_gpu


/git/py/env/rl0d/bin/python /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 --no_gpu

/git/py/env/rl0d/bin/python /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n5_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 --no_gpu
```

# q2
```shell

/git/py/env/rl0d/bin/python /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/run_hw4_mb.py --exp_name q2_obstacles_singleiteration --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10
```



# q3
```shell

/git/py/env/rl0d/bin/python /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/run_hw4_mb.py --exp_name q3_cheetah --env_name cheetah-cs285-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 20

/git/py/env/rl0d/bin/python /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/run_hw4_mb.py --exp_name q3_obstacles --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --n_iter 12

/git/py/env/rl0d/bin/python /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/run_hw4_mb.py --exp_name q3_reacher --env_name reacher-cs285-va0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 5000 --n_iter 15
```