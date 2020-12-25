## last last run:
### 11-23
### q2
    tb /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/../../data/hw4_q2_obstacles_singleiteration_obstacles-cs285-v0_23-11-2020_15-15-04

You can expect TrainAverageReturn to be around -160 and EvalAverageReturn to be around -70 to -50.

    **Eval_AverageReturn** : -30.09004783630371
    **Train_AverageReturn** : -175.22740173339844

run 2
 
    tb  /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/../../data/hw4_q2_obstacles_singleiteration_obstacles-cs285-v0_23-11-2020_15-22-27
    Eval_AverageReturn : -31.481815338134766 
    Train_AverageReturn : -171.75384521484375

### q3
### cheetah & reacher
    tb  /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/../../data/hw4_q3_cheetah_cheetah-cs285-v0_23-11-2020_11-32-03  # PARTIAL. expect rewards of around 250-350 for the cheetah env takes 3-4 hours. got Train_AverageReturn : 327.07977294921875 @ Iteration 9
    tb  /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/../../data/hw4_q3_reacher_reacher-cs285-v0_23-11-2020_07-29-42  # GOOD, got -266.8, expect -250 to -300 for the reacher env
#### obstacles  (a litter inferior to expected)
    tb  /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/../../data/hw4_q3_obstacles_obstacles-cs285-v0_23-11-2020_07-29-48  # -25.9 at best
    tb  /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/../../data/hw4_q3_obstacles_obstacles-cs285-v0_23-11-2020_11-33-09  # gpu: -26.73

### q4 cmds: (all success, do not run any more because they are exactly enough for inspection)
sample path:
/git/py.code/hw4/homework_fall2020/hw4/data/hw4_q4_reacher_horizon30_reacher-cs285-v0_23-11-2020_07-36-54
    
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon5 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 5 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon15 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_horizon30 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 30 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
    
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_numseq100 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 100
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_numseq1000 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 1000
    
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble1 --env_name reacher-cs285-v0 --ensemble_size 1 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble3 --env_name reacher-cs285-v0 --ensemble_size 3 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
    python cs285/scripts/run_hw4_mb.py --exp_name q4_reacher_ensemble5 --env_name reacher-cs285-v0 --ensemble_size 5 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15

    #->



### 11-22
partial: 
 /git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/../../data/hw4_q3_reacher_reacher-cs285-v0_22-11-2020_20-15-10 
/git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/../../data/hw4_q3_cheetah_cheetah-cs285-v0_22-11-2020_19-50-36 


full:
/git/py.code/hw4/homework_fall2020/hw4/cs285/scripts/../../data/hw4_q3_obstacles_obstacles-cs285-v0_22-11-2020_19-50-38  # got -31.83, expect -25 ~ -20



## Errors when parallel + GPU
    RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
      File "/git/py.code/hw4/homework_fall2020/hw4/cs285/policies/MPC_policy.py", line 69, in get_action
    obs, candidate_action_sequences, model)
  File "/git/py.code/hw4/homework_fall2020/hw4/cs285/policies/MPC_policy.py", line 106, in calculate_sum_of_rewards
    predicted_obs_after_step_i = model.get_prediction(obs_batch, action_batch, self.data_statistics)
  File "/git/py.code/hw4/homework_fall2020/hw4/cs285/models/ff_model.py", line 125, in get_prediction
    data_statistics['delta_mean'], data_statistics['delta_std'])
  File "/git/py.code/hw4/homework_fall2020/hw4/cs285/models/ff_model.py", line 101, in forward
    delta_pred_normalized = self.delta_network(concatenated_input)
  File "/git/py/env/rl0d/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/git/py/env/rl0d/lib/python3.6/site-packages/torch/nn/modules/container.py", line 117, in forward
    input = module(input)
  File "/git/py/env/rl0d/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/git/py/env/rl0d/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 91, in forward
    return F.linear(input, self.weight, self.bias)
  File "/git/py/env/rl0d/lib/python3.6/site-packages/torch/nn/functional.py", line 1674, in linear
    ret = torch.addmm(bias, input, weight.t())
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`


    
    File "/git/py.code/hw4/homework_fall2020/hw4/cs285/infrastructure/utils.py", line 237, in unnormalize
    return data * std + mean
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
