from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

from cs285.infrastructure import pytorch_util as ptu


class DQNCritic(BaseCritic):

  def __init__(self, hparams, optimizer_spec, **kwargs):
    super().__init__(**kwargs)
    self.env_name = hparams['env_name']
    self.ob_dim = hparams['ob_dim']

    if isinstance(self.ob_dim, int):
      self.input_shape = (self.ob_dim,)
    else:
      self.input_shape = hparams['input_shape']

    self.ac_dim = hparams['ac_dim']
    self.double_q = hparams['double_q']
    self.grad_norm_clipping = hparams['grad_norm_clipping']
    self.gamma = hparams['gamma']

    self.optimizer_spec = optimizer_spec
    network_initializer = hparams['q_func']
    self.q_net = network_initializer(self.ob_dim, self.ac_dim)
    self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
    self.optimizer = self.optimizer_spec.constructor(
      self.q_net.parameters(),
      **self.optimizer_spec.optim_kwargs
    )
    self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
      self.optimizer,
      self.optimizer_spec.learning_rate_schedule,
    )
    self.loss = nn.SmoothL1Loss()  # AKA Huber loss
    self.q_net.to(ptu.device)
    self.q_net_target.to(ptu.device)

  def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    """
        Update the parameters of the critic.
        let sum_of_path_lengths be the sum of the lengths of the paths sampled from
            Agent.sample_trajectories
        let num_paths be the number of paths sampled from Agent.sample_trajectories
        arguments:
            ob_no: shape: (sum_of_path_lengths, ob_dim)
            next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
            reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                the reward for each timestep
            terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                at that timestep of 0 if the episode did not end
        returns:
            nothing
    """
    ob_no = ptu.from_numpy(ob_no)
    ac_na = ptu.from_numpy(ac_na).to(torch.long)
    next_ob_no = ptu.from_numpy(next_ob_no)
    reward_n = ptu.from_numpy(reward_n)
    terminal_n = ptu.from_numpy(terminal_n)

    qa_t_values = self.q_net(ob_no)
    # gather() for 3d tensor:
    # out[i][j][k] = input[i][ index[i][j][k] ] [k]  # if dim == 1
    q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)

    # TODO compute the Q-values from the target network
    qa_tp1_values = self.q_net_target(next_ob_no)
    if self.double_q:
      # You must fill this part for Q2 of the Q-learning portion of the homework.
      # In double Q-learning, the best action is selected using the Q-network that
      # is being updated, but the Q-value for this action is obtained from the
      # target Q-network. See page 5 of https://arxiv.org/pdf/1509.06461.pdf for more details.
      q_phi_s_prime_a_prime = self.q_net(next_ob_no) # (32,6)
      _, argmax = q_phi_s_prime_a_prime.max(dim=1) #  get max of every row.  (32,)
      # q_phi_prime_of_argmax_q_phi = qa_tp1_values[argmax] # torch.Size([32, 6])[(32,)] -> torch.Size([32, 6])
      q_phi_prime_of_argmax_q_phi = qa_tp1_values.gather(1, argmax.view(-1,1)) # shape: torch.Size([32, 1])
      q_phi_prime_of_argmax_q_phi = q_phi_prime_of_argmax_q_phi.squeeze()
      q_tp1 = q_phi_prime_of_argmax_q_phi # rename
      '''
      `gather` example:
      m = torch.randn(4,2)
      tensor([[ 1.2593,  1.1184],
        [ 1.1371,  0.5643],
        [-2.0850,  1.5010],
        [ 0.1900,  0.1847]])
      ids = torch.Tensor([1,1,0,0]).long()
      tensor([[ 1.2593,  1.1184],
        [ 1.1371,  0.5643],
        [-2.0850,  1.5010],
        [ 0.1900,  0.1847]])
      print(m.gather(1, ids.view(-1,1)))
      tensor([[ 1.1184],
        [ 0.5643],
        [-2.0850],
        [ 0.1900]])
      '''

    else:
      q_tp1, _ = qa_tp1_values.max(dim=1) # one example shape of qa_tp1_values: torch.Size([32, 6]). get max of every row, q_tp1 shape: 32

    # TODO compute targets for minimizing Bellman error
    # HINT: as you saw in lecture, this would be:
    # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
    terminal_n = terminal_n.bool()
    not_terminal_n = ~terminal_n
    not_terminal_n = not_terminal_n.float()
    target = reward_n + self.gamma * q_tp1 * not_terminal_n
    target = target.detach()

    assert q_t_values.shape == target.shape
    loss = self.loss(q_t_values, target)

    self.optimizer.zero_grad()
    loss.backward()
    utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
    self.optimizer.step()

    return {
      'Training Loss': ptu.to_numpy(loss),
    }

  def update_target_network(self):
    for target_param, param in zip(
        self.q_net_target.parameters(), self.q_net.parameters()
    ):
      target_param.data.copy_(param.data)

  def qa_values(self, obs):
    obs = ptu.from_numpy(obs)
    qa_values = self.q_net(obs)
    return ptu.to_numpy(qa_values)
