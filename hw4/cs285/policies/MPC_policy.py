import numpy as np
from cs285.models.ff_model import FFModel

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

  def __init__(self,
               env,
               ac_dim,
               dyn_models,
               horizon,
               N,
               **kwargs
               ):
    super().__init__(**kwargs)

    # init vars
    self.env = env
    self.dyn_models = dyn_models
    self.horizon = horizon
    self.N = N
    self.data_statistics = None  # NOTE must be updated from elsewhere

    self.ob_dim = self.env.observation_space.shape[0]

    # action space
    self.ac_space = self.env.action_space
    self.ac_dim = ac_dim
    self.low = self.ac_space.low
    self.high = self.ac_space.high

  def sample_action_sequences(self, num_sequences, horizon):
    '''
    return     random_action_sequences
    candidate_action_sequences: numpy array with the candidate action
      sequences. Shape [N, H, D_action] where
          - N is the number of action sequences considered
          - H is the horizon
          - D_action is the action of the dimension
    '''
    # TODO(Q1) uniformly sample trajectories and return an array of
    # dimensions (num_sequences, horizon, self.ac_dim) in the range
    # [self.low, self.high]
    random_action_sequences = np.zeros(shape=(num_sequences, horizon, self.ac_dim))

    for i in range(num_sequences):
      for j in range(horizon):
        if num_sequences != 1 and horizon != 1:
          print(end='')
        random_action_sequences[i][j] = self.ac_space.sample()
    return random_action_sequences

  def get_action(self, obs):

    if self.data_statistics is None:
      # print("WARNING: performing random actions.")
      return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

    # sample random actions (N x horizon)
    candidate_action_sequences = self.sample_action_sequences(
      num_sequences=self.N, horizon=self.horizon)

    # for each model in ensemble:
    predicted_sum_of_rewards_per_model = []
    for model in self.dyn_models:
      sum_of_rewards = self.calculate_sum_of_rewards(
        obs, candidate_action_sequences, model)
      predicted_sum_of_rewards_per_model.append(sum_of_rewards)

    # calculate mean_across_ensembles(predicted rewards)
    predicted_rewards = np.mean(
      predicted_sum_of_rewards_per_model, axis=0)  # [ens, N] --> N

    # pick the action sequence and return the 1st element of that sequence
    best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)]
    action_to_take = best_action_sequence[0]
    return action_to_take[None]  # Unsqueeze the first index

  def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model: FFModel):
    """
    :param obs: numpy array with the *current observation*. Shape [D_obs]
    :param candidate_action_sequences: numpy array with the candidate action
    sequences. Shape [N, H, D_action] where
        - N is the number of action sequences considered
        - H is the horizon
        - D_action is the action of the dimension
    :param model: The current dynamics model.
    :return: numpy array with the sum of rewards for each action sequence.
    The array should have shape [N].
    """
    N = candidate_action_sequences.shape[0]
    H = candidate_action_sequences.shape[1]
    # For each candidate action sequence, predict a sequence of
    # states for each dynamics model in your ensemble.
    predicted_obs = np.zeros(shape=(N, H, obs.shape[0]))
    for step_i in range(H):  # iterate in one step in horizon H
      # x=array([1, 2]); np.repeat(x[np.newaxis,:],3,axis=0)--> array([[1, 2],[1, 2],[1, 2]])
      if step_i==0:
        obs_batch = np.repeat(obs[np.newaxis, :], N, axis=0)
      else:
        obs_batch = predicted_obs_after_step_i
      action_batch = candidate_action_sequences[:, step_i, :]
      assert action_batch.shape[0] == obs_batch.shape[0]
      predicted_obs_after_step_i = model.get_prediction(obs_batch, action_batch, self.data_statistics)
      assert predicted_obs[:, step_i, :].shape == predicted_obs_after_step_i.shape
      predicted_obs[:, step_i, :] = predicted_obs_after_step_i

    # Once you have a sequence of predicted states from each model in
    # your ensemble, calculate the sum of rewards for each sequence
    # using `self.env.get_reward(predicted_obs)`
    '''self.env.get_reward
    Args:
        observations: (batchsize, obs_dim) or (obs_dim,)
        actions: (batchsize, ac_dim) or (ac_dim,)
    Return:
        r_total: reward of this (o,a) pair, dimension is (batchsize,1) or (1,)
        done: True if env reaches terminal state, dimension is (batchsize,1) or (1,)
    '''
    sum_of_rewards = np.zeros(shape=N)
    for action_sequence_i in range(N):
      observations = predicted_obs[action_sequence_i]
      actions = candidate_action_sequences[action_sequence_i]
      # TODO check whether `done` needs to be used?
      r_total_list, _ = self.env.get_reward(observations, actions) # check whether r_total is a batch?
      sum_of_rewards[action_sequence_i] = sum(r_total_list)
    # You should sum across `self.horizon` time step.
    # Hint: you should use model.get_prediction and you shouldn't need
    #       to import pytorch in this file.
    # Hint: Remember that the model can process observations and actions
    #       in batch, which can be much faster than looping through each
    #       action sequence.
    return sum_of_rewards
