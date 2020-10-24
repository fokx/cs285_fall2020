from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import glob

logdirs = [
  '/git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q4_ac_1_1-ntu_CartPole-v0_21-10-2020_20-04-38',
  '/git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q4_100_1_CartPole-v0_21-10-2020_20-08-47',
  '/git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q4_1_100_CartPole-v0_21-10-2020_20-08-57',
  '/git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q4_10_10_CartPole-v0_21-10-2020_20-04-35'
]
logdirs = [i + "/events*" for i in logdirs]

for logdir in logdirs:
  eventfile = glob.glob(logdir)[0]
  X, Y = [], []
  for iterator in summary_iterator(eventfile):
    for v in iterator.summary.value:
      if v.tag == 'Train_EnvstepsSoFar':
        X.append(v.simple_value)
      elif v.tag == 'Eval_AverageReturn':
        Y.append(v.simple_value)
    print()
