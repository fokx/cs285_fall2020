import os, shutil, subprocess
from subprocess import Popen

# for b in [10000,30000,50000]:
#   for lr in [0.005,0.01,0.02]:
#     print(base_str.format(b=b,lr=lr))

base_dir = "/homework_fall2020/hw2/data/"
ff_cmds = []
for port_num, sub_dir in enumerate(os.listdir(base_dir)):
  port_num += 7000
  ff_cmd = "firefox http://127.0.0.1:{}".format(port_num)
  ff_cmds.append(ff_cmd)

processes_ff = [Popen(cmd, shell=True) for cmd in ff_cmds]
for p in processes_ff:
  p.wait()
