#import gym
import numpy as np
import itertools
list_of_list=[[1,2],[7,9,0]]
d=np.concatenate(list_of_list)
e= itertools.chain([1,2],[7,9,0])
e2= itertools.chain(*list_of_list)
e=list(e)
e2=list(e2)
#
# env = gym.make("InvertedPendulum-v2")
# ac = env.action_space.sample()
# ob = env.observation_space.sample()
# env.reset()
# observation, reward, done, info = env.step(ac)
print()
