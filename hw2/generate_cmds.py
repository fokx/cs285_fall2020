import os, shutil, glob
import  subprocess
from subprocess import Popen
import pprint
base_str = "python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {b} -lr {lr} -rtg --nn_baseline --exp_name q4_search_b{b}_lr{lr}_rtg_nnbaseline"
# for b in [10000,30000,50000]:
#   for lr in [0.005,0.01,0.02]:
#     print(base_str.format(b=b,lr=lr))

base_dir = "/homework_fall2020/hw2/data/"
# for i in os.listdir(base_dir):
#   print("tensorboard --logdir  " + os.path.join(base_dir, i))
cheetah_dirs = glob.glob('/git/py.code/hw2/homework_fall2020/hw2/data/q4_search*')
for cheetah_dir in cheetah_dirs:
  # os.system("tensorboard --logdir  {}".format(cheetah_dir))
  print(end='')
tb_cmds = ["tensorboard --logdir  {}  --port {}".format(dirname, i+7100) for i, dirname in enumerate(cheetah_dirs)]
processes_tb = [Popen(cmd, shell=True) for cmd in tb_cmds]
pprint.pprint(tb_cmds)
for p in processes_tb:
  p.wait()
'''
python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.005_rtg_nnbaseline
python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.01_rtg_nnbaseline
python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.02_rtg_nnbaseline
python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.005_rtg_nnbaseline
python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.01_rtg_nnbaseline
python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.02_rtg_nnbaseline
python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.005_rtg_nnbaseline
python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.01 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.01_rtg_nnbaseline
python /git/py.code/hw2/homework_fall2020/hw2/cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.02_rtg_nnbaseline

'''


'''
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-12-18
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q1_sb_rtg_dsa_CartPole-v0_20-10-2020_13-59-47
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-03-47
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q2_b5000_r0.01_InvertedPendulum-v2_20-10-2020_14-00-03
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-26-54
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q1_sb_rtg_na_CartPole-v0_20-10-2020_14-01-14
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_15-10-46
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q1_lb_rtg_dsa_CartPole-v0_20-10-2020_14-07-28
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-12-30
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-26-33
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q1_lb_rtg_na_CartPole-v0_20-10-2020_14-14-13
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-12-03
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q1_lb_no_rtg_dsa_CartPole-v0_20-10-2020_14-02-45
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-12-26
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q1_sb_no_rtg_dsa_CartPole-v0_20-10-2020_13-52-03
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q3_b40000_r0.005_LunarLanderContinuous-v2_20-10-2020_14-02-38
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_20-10-2020_14-49-51
tensorboard --logdir  /git/py.code/hw2/homework_fall2020/hw2/data/q1_sb_no_rtg_dsa_CartPole-v0_20-10-2020_13-58-46
'''
