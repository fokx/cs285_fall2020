import glob
import tensorflow.compat.v1 as tf


def get_section_results(file):
  """
      requires tensorflow==1.12.0
  """
  X = []
  Y = []
  for e in tf.train.summary_iterator(file):
    for v in e.summary.value:
      if v.tag == 'Train_EnvstepsSoFar':
        X.append(v.simple_value)
      elif v.tag == 'Eval_AverageReturn':
        Y.append(v.simple_value)
  return X, Y


if __name__ == '__main__':
  import glob

  logdir = '/git/py.code/hw4/homework_fall2020/hw4/data/hw4_q1_cheetah_n5_arch2x250_cheetah-cs285-v0_09-11-2020_12-00-21/events*'
  eventfile = glob.glob(logdir)[0]

  X, Y = get_section_results(eventfile)
  for i, (x, y) in enumerate(zip(X, Y)):
    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
