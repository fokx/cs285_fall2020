import tensorflow.compat.v1 as tf  # pycharm says there is no compat module in tensorflow, which is wrong
import matplotlib.pyplot as plt
import glob


def get_section_results(file):
  """
      requires tensorflow==1.12.0
  """
  X = []
  Y = []
  # for e in tf.data.TFRecordDataset(file):
  #   print(e)
  for e in tf.train.summary_iterator(file):
    for v in e.summary.value:
      if v.tag == 'Train_EnvstepsSoFar':
        X.append(v.simple_value)
      elif v.tag == 'Eval_AverageReturn':
        Y.append(v.simple_value)
  return X, Y


def get_X_Y(logdir, line_width):
  eventfile = glob.glob(logdir)[0]

  X, Y = get_section_results(eventfile)
  # X_Y = zip(X, Y)
  X_Y = [X, Y]
  plt.plot(X, Y, linewidth=line_width)
  plt.show()
  # for i, (x, y) in enumerate(zip(X, Y)):
  #   print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
  return X_Y


if __name__ == '__main__':


  logdir = '/git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q4_ac_1_1-ntu_CartPole-v0_21-10-2020_20-04-38/events*'
  logdirs = [
    '/git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q4_ac_1_1-ntu_CartPole-v0_21-10-2020_20-04-38',
    '/git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q4_100_1_CartPole-v0_21-10-2020_20-08-47',
    '/git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q4_1_100_CartPole-v0_21-10-2020_20-08-57',
    '/git/py.code/hw3/homework_fall2020/hw3/cs285/scripts/../data/hw3_q4_10_10_CartPole-v0_21-10-2020_20-04-35'
  ]
  logdirs = [i + "/events*" for i in logdirs]
  X_Y_list = []
  line_width = 0.5
  for logdir in logdirs:
    print(line_width)
    X_Y_list.append(get_X_Y(logdir, line_width=line_width))
    line_width += 0.5
  plt.show()
