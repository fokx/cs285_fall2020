import numpy as np
import pdb
from cs285.infrastructure import pytorch_util as ptu


class ArgMaxPolicy(object):

  def __init__(self, critic):
    # critic: <cs285.critics.dqn_critic.DQNCritic object at 0x7f68ccc8e7b8>
    self.critic = critic

  def set_critic(self, critic):
    self.critic = critic

  def get_action(self, obs):
    if len(obs.shape) > 3:
      observation = obs
    else:
      observation = obs[None]

    ## return the action that maxinmizes the Q-value
    # at the current observation as the output
    observation = ptu.from_numpy(observation)
    action_values = self.critic.q_net(observation)
    # action_values shape: torch.Size([1, 1, 6])
    # max_action_value1, action1 = action_values.max(dim=-1) # unpack (max, argmax)
    max_action_value, action = action_values.max(dim=1) # unpack (max, argmax)

    return int(action.squeeze().detach().cpu().numpy())
    # return action

