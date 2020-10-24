import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

  def __init__(self,
               ac_dim,
               ob_dim,
               n_layers,
               size,
               discrete=False,
               learning_rate=1e-4,
               training=True,
               nn_baseline=False,
               **kwargs
               ):
    super().__init__(**kwargs)

    # init vars
    self.ac_dim = ac_dim
    self.ob_dim = ob_dim
    self.n_layers = n_layers
    self.discrete = discrete
    self.size = size
    self.learning_rate = learning_rate
    self.training = training
    self.nn_baseline = nn_baseline

    if self.discrete:
      self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                     output_size=self.ac_dim,
                                     n_layers=self.n_layers,
                                     size=self.size)
      self.logits_na.to(ptu.device)
      self.mean_net = None
      self.logstd = None
      self.optimizer = optim.Adam(self.logits_na.parameters(),
                                  self.learning_rate)
    else:
      self.logits_na = None
      self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                    output_size=self.ac_dim,
                                    n_layers=self.n_layers, size=self.size)
      self.logstd = nn.Parameter(
        torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
      )
      self.mean_net.to(ptu.device)
      self.logstd.to(ptu.device)
      self.optimizer = optim.Adam(
        itertools.chain([self.logstd], self.mean_net.parameters()),
        self.learning_rate
      )

    if nn_baseline:
      self.baseline = ptu.build_mlp(
        input_size=self.ob_dim,
        output_size=1,
        n_layers=self.n_layers,
        size=self.size,
      )
      self.baseline.to(ptu.device)
      self.baseline_optimizer = optim.Adam(
        self.baseline.parameters(),
        self.learning_rate,
      )
    else:
      self.baseline = None

  ##################################

  def save(self, filepath):
    torch.save(self.state_dict(), filepath)

  ##################################

  # query the policy with observation(s) to get selected action(s)
  def get_action(self, obs: np.ndarray) -> np.ndarray:
    if len(obs.shape) > 1:
      observation = obs
    else:
      observation = obs[None]  # make a minibatch, if a=array([1, 2]), then a[None]=array([[1, 2]])
    # return the action that the policy prescribes
    if self.discrete:
      observation = ptu.from_numpy(observation)
      action_dist = self.forward(observation)
      action = action_dist.sample()
      # # action_2 = action_2[0]
      # # return int(action)
      # # return action
      # forward_pass = self.logits_na
      # action_prob = forward_pass(observation)
      # # action_dist = distributions.Categorical(probs=action_prob)
      # action2 = torch.argmax(action_prob)
      # action2 = action2[None]
      return action
    else:
      action_dist = self.forward(ptu.from_numpy(observation))
      action = action_dist.sample()
      # action = action[None]
      return action

  # update/train this policy
  def update(self, observations, actions, **kwargs):
    raise NotImplementedError

  # This function defines the forward pass of the network.
  # You can return anything you want, but you should be able to differentiate
  # through it. For example, you can return a torch.FloatTensor. You can also
  # return more flexible objects, such as a
  # `torch.distributions.Distribution` object. It's up to you!
  def forward(self, observation: torch.FloatTensor):
    # observation = torch.from_numpy(observation.astype(np.float32))
    if self.discrete:
      action_logits = self.logits_na(observation)
      action_dist = distributions.Categorical(logits=action_logits)
      # action = torch.argmax(action)
      # action = action[None]
      return action_dist
    else:
      mean = self.mean_net(observation)
      mean = mean.squeeze(dim=-1)
      # mean = mean[0]
      logstd = self.logstd
      # logstd = logstd[0]
      # action_distribution = distributions.MultivariateNormal(mean, scale_tril=torch.diag(logstd))
      action_distribution = distributions.Normal(loc=mean,
                                                 scale=torch.exp(logstd))  # will the scale overflow action space?
      return action_distribution
    # return action.detach().numpy()
    # return action_distribution


#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
  def update(self, observations, actions, adv_n=None):
    # TODO: update the policy and return the loss
    # observations = ptu.from_numpy(observations)
    # actions = ptu.from_numpy(actions)
    if adv_n is not None:
      # adv_n = ptu.from_numpy(adv_n)
      pass
    else:
      # in which circumstances can adv_n be None?? seems no
      raise ValueError("adv_n is None!?")
    action_dist = self.forward(observations)
    if self.discrete:
      log_pi = action_dist.log_prob(actions)
    else:
      if len(action_dist.batch_shape) == 1:
        log_pi = action_dist.log_prob(actions)
      else:
        action_dist_new = distributions.Independent(action_dist, 1)
        log_pi = action_dist_new.log_prob(actions)
    assert adv_n.ndim == log_pi.ndim
    sums = adv_n * log_pi
    # sums = torch.tensor(sums)l
    # loss = sum(sums)
    loss = -torch.sum(sums)  # `optimizer.step()` MINIMIZES a loss but we want to MAXIMIZE expectation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item() # what  does item() do
