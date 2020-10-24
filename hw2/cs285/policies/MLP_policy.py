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
from cs285.infrastructure import utils


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
      # TODO squeeze logstd
      # logstd = logstd[0]
      # action_distribution = distributions.MultivariateNormal(mean, scale_tril=torch.diag(logstd))
      action_distribution = distributions.Normal(loc=mean,
                                                 scale=torch.exp(logstd))  # will the scale overflow action space?
      return action_distribution
    # return action.detach().numpy()
    # return action_distribution


#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
  def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
    super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
    self.baseline_loss = nn.MSELoss()

  def update(self, observations, actions, advantages, q_values=None):
    observations = ptu.from_numpy(observations)
    actions = ptu.from_numpy(actions)
    advantages = ptu.from_numpy(advantages)

    # TODO done: compute the loss that should be optimized when training with policy gradient
    # HINT1: Recall that the expression that we want to MAXIMIZE
    # is the expectation over collected trajectories of:
    # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
    # HINT2: you will want to use the `log_prob` method on the distribution returned
    # by the `forward` method
    # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss

    action_dist = self.forward(observations)
    if self.discrete:
      log_pi = action_dist.log_prob(actions)
    else:
      # distributions.Independent:
      # Reinterprets some of the batch dims of a distribution as event dims.
      # This is mainly useful for changing the shape of the result of log_prob.

      """from the experience from debugging,
      for lunarLander, action_dist.batch_shape = [5004]
        -> use it directly
      for invertedPendulum, action_dist.batch_shape = torch.Size([40070, 2]) 
        -> use action_dist_new, whose batch_shape = 40070
      """
      if len(action_dist.batch_shape) == 1:
        log_pi = action_dist.log_prob(actions)
      else:
        action_dist_new = distributions.Independent(action_dist, 1)
        log_pi = action_dist_new.log_prob(actions)

    # sums = [entry * adv for entry in log_pi for adv in advantages]
    # sums = ptu.from_numpy(sums)
    # log pi can be inf if using multivariate normal
    # what if log_pi element size is not 1?
    # sums = torch.mul(log_pi, advantages)  # ? is it the same as below? -- high chances that they are the same
    assert advantages.ndim == log_pi.ndim
    sums = advantages * log_pi
    # sums = torch.tensor(sums)
    # loss = sum(sums)
    loss = -torch.sum(sums)  # `optimizer.step()` MINIMIZES a loss but we want to MAXIMIZE expectation

    # TODO done: optimize `loss` using `self.optimizer`
    # HINT: remember to `zero_grad` first
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    '''
       File "/home/hawk/dl/cs285/hw2/homework_fall2020/hw2/cs285/policies/MLP_policy.py", line 172, in update
   sums = torch.mul(log_pi, advantages) # ? is it the same as below?
   RuntimeError: The size of tensor a (2) must match the size of tensor b (40006) at non-singleton dimension 1
   logged outputs to  /home/hawk/dl/cs285/hw2/homework_fall2020/hw2/cs285/scripts/../../data/q3_b40000_r0.005_LunarLanderContinuous-v2_21-09-2020_10-16-04
   '''
    if self.nn_baseline:
      ## TODO: normalize the q_values to have a mean of zero and a standard deviation of one
      ## HINT: there is a `normalize` function in `infrastructure.utils`
      targets = utils.normalize(q_values, np.mean(q_values), np.std(q_values))
      targets = ptu.from_numpy(targets)

      ## TODO: use the `forward` method of `self.baseline` to get baseline predictions
      baseline_predictions = self.baseline.forward(observations)

      ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
      ## [ N ] versus shape [ N x 1 ]
      ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
      # TODO ? move squeeze into model.forward
      baseline_predictions = baseline_predictions.squeeze()
      assert baseline_predictions.shape == targets.shape, "{} vs {}".format(baseline_predictions.shape,
                                                                            targets.shape)

      # TODO: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
      # HINT: use `F.mse_loss`
      baseline_loss = F.mse_loss(baseline_predictions, targets)

      # TODO: optimize `baseline_loss` using `self.baseline_optimizer`
      # HINT: remember to `zero_grad` first
      self.baseline_optimizer.zero_grad()
      baseline_loss.backward()
      self.baseline_optimizer.step()

    train_log = {
      'Training Loss': ptu.to_numpy(loss),
    }
    return train_log

  def run_baseline_prediction(self, obs):
    """
        Helper function that converts `obs` to a tensor,
        calls the forward method of the baseline MLP,
        and returns a np array

        Input: `obs`: np.ndarray of size [N, 1]
        Output: np.ndarray of size [N]

    """
    obs = ptu.from_numpy(obs)
    predictions = self.baseline(obs)
    return ptu.to_numpy(predictions)[:, 0]
