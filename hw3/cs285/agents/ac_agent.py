from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
  BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
from cs285.infrastructure import pytorch_util as ptu
import torch

class ACAgent(BaseAgent):
  def __init__(self, env, agent_params):
    super(ACAgent, self).__init__()

    self.env = env
    self.agent_params = agent_params

    self.gamma = self.agent_params['gamma']
    self.standardize_advantages = self.agent_params['standardize_advantages']

    self.actor = MLPPolicyAC(
      self.agent_params['ac_dim'],
      self.agent_params['ob_dim'],
      self.agent_params['n_layers'],
      self.agent_params['size'],
      self.agent_params['discrete'],
      self.agent_params['learning_rate'],
    )
    self.critic = BootstrappedContinuousCritic(self.agent_params)

    self.replay_buffer = ReplayBuffer()

  def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
    ob_no = ptu.from_numpy(ob_no)
    next_ob_no = ptu.from_numpy(next_ob_no)
    terminal_n = ptu.from_numpy(terminal_n)
    re_n = ptu.from_numpy(re_n)

    ac_na = ptu.from_numpy(ac_na)

    loss_critic = 0.
    for i in range(self.agent_params['num_critic_updates_per_agent_update']):
      loss_critic += self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

    # advantage = estimate_advantage(...) :
    adv_n = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n) # a tensor is returned
    loss_actor = 0.
    for i in range(self.agent_params['num_actor_updates_per_agent_update']):
      loss_actor += self.actor.update(ob_no, ac_na, adv_n)

    loss = OrderedDict()
    loss['Critic_Loss'] = loss_critic
    loss['Actor_Loss'] = loss_actor # in TensorBoard, loss_actor actually increases as we actually minimize -loss_actor

    return loss

  def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
    # TODO Implement the following pseudocode:
    # 1) query the critic with ob_no, to get V(s)
    # 2) query the critic with next_ob_no, to get V(s')
    # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
    # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
    # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)


    # V_s_prime = self.critic.critic_network(next_ob_no)
    # V_s_prime = V_s_prime.squeeze()
    # mask = (terminal_n == 1.)
    # V_s_prime= V_s_prime.masked_fill(mask, 0.)
    #
    # V_s = self.critic.critic_network(ob_no)
    # V_s = V_s.squeeze()
    # # assert V_s_prime.ndim == V_s.ndim     # TODO-assert enable this assert in debug
    # adv_n2 = re_n + self.gamma * V_s_prime - V_s

    # another way to calculate:
    V_s_prime = re_n + (1 - terminal_n) * self.gamma * self.critic.forward(next_ob_no)
    adv_n = V_s_prime - self.critic.forward(ob_no)
    # assert adv_n2 == adv_n

    if self.standardize_advantages:
      adv_n = (adv_n - torch.mean(adv_n)) / (torch.std(adv_n) + 1e-8)
    return adv_n

  def add_to_replay_buffer(self, paths):
    self.replay_buffer.add_rollouts(paths)

  def sample(self, batch_size):
    return self.replay_buffer.sample_recent_data(batch_size)
