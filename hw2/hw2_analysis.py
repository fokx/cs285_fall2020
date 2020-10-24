import os, shutil, subprocess
from subprocess import Popen

base_str = "python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {b} -lr {lr} -rtg --nn_baseline --exp_name q4_search_b{b}_lr{lr}_rtg_nnbaseline"
# for b in [10000,30000,50000]:
#   for lr in [0.005,0.01,0.02]:
#     print(base_str.format(b=b,lr=lr))

base_dir = "/homework_fall2020/hw2/data/"
tb_cmds = []
# ff_cmds = []
for port_num, sub_dir in enumerate(os.listdir(base_dir)):
  port_num += 7000
  tb_cmd = "tensorboard --logdir {} --port {} ".format(os.path.join(base_dir, sub_dir), port_num)
  ff_cmd = "firefox http://127.0.0.1:{}".format(port_num)
  print(tb_cmd)
  tb_cmds.append(tb_cmd)
  # ff_cmds.append(ff_cmd)

# run in parallel
processes_tb = [Popen(cmd, shell=True) for cmd in tb_cmds]
# processes_ff = [Popen(cmd, shell=True) for cmd in ff_cmds]
# wait for completion
for p in processes_tb:
  p.wait()
