import numpy as np
import time
import copy


############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):
  model = models[0]

  # true
  true_states = perform_actions(env, action_sequence)['observation']

  # predicted
  ob = np.expand_dims(true_states[0], 0)
  pred_states = []
  for ac in action_sequence:
    pred_states.append(ob)
    action = np.expand_dims(ac, 0)
    ob = model.get_prediction(ob, action, data_statistics)
  pred_states = np.squeeze(pred_states)

  # mpe
  mpe = mean_squared_error(pred_states, true_states)

  return mpe, true_states, pred_states


def perform_actions(env, actions):
  ob = env.reset()
  obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
  steps = 0
  for ac in actions:
    obs.append(ob)
    acs.append(ac)
    ob, rew, done, _ = env.step(ac)
    # add the observation after taking a step to next_obs
    next_obs.append(ob)
    rewards.append(rew)
    steps += 1
    # If the episode ended, the corresponding terminal value is 1
    # otherwise, it is 0
    if done:
      terminals.append(1)
      break
    else:
      terminals.append(0)

  return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def mean_squared_error(a, b):
  return np.mean((a - b) ** 2)


############################################
############################################
def sample_trajectory(env, policy, max_path_length, render=False, render_mode='rgb_array'):
  # initialize env for the beginning of a new rollout
  ob = env.reset()  # HINT: should be the output of resetting the env

  # init vars
  obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
  steps = 0
  while True:

    # render image of the simulated env
    if render:
      if 'rgb_array' in render_mode:
        if hasattr(env, 'sim'):
          image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
        else:
          image_obs.append(env.render(mode=render_mode))
      if 'human' in render_mode:
        env.render(mode=render_mode)
        time.sleep(env.model.opt.timestep)

    # use the most recent ob to decide what to do
    obs.append(ob)

    ac = policy.get_action(ob)  # HINT: query the policy's get_action function
    ac = ac[0].cpu().detach().numpy()
    # ac = ac[0]
    # ac = int(ac) # explicitly converted to int is only applicable to discrete action space
    acs.append(ac)
    # take that action and record results
    ob, rew, done, _ = env.step(ac)  # _ = info

    # record result of taking that action
    steps += 1
    next_obs.append(ob)
    rewards.append(rew)

    # HINT: rollout can end due to done, or due to max_path_length
    rollout_done = False  # HINT: this is either 0 or 1
    if done or steps >= max_path_length:
      rollout_done = True
    terminals.append(rollout_done)

    if rollout_done:
      break

  # obs lags one element to next_obs
  return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectory_NOCUDA(mp_timesteps_this_batch, min_timesteps_per_batch, mp_paths, enough_event,
                      env, policy, max_path_length, render=False, render_mode=('rgb_array')):
  while True:
    # initialize env for the beginning of a new rollout
    ob = env.reset()  # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
      # render image of the simulated env
      if render:
        if 'rgb_array' in render_mode:
          if hasattr(env, 'sim'):
            image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
          else:
            image_obs.append(env.render(mode=render_mode))
        if 'human' in render_mode:
          env.render(mode=render_mode)
          time.sleep(env.model.opt.timestep)

      # use the most recent ob to decide what to do
      obs.append(ob)

      ac = policy.get_action(ob)  # HINT: query the policy's get_action function
      ac = ac[0].cpu().detach().numpy()
      # ac = int(ac) # explicitly converted to int is only applicable to discrete action space
      acs.append(ac)
      # take that action and record results
      ob, rew, done, _ = env.step(ac)  # _ = info

      # record result of taking that action
      steps += 1
      next_obs.append(ob)
      rewards.append(rew)

      # HINT: rollout can end due to done, or due to max_path_length
      rollout_done = False  # HINT: this is either 0 or 1
      if done or steps >= max_path_length:
        rollout_done = True
      terminals.append(rollout_done)

      if rollout_done:
        break

    # obs lags one element to next_obs
    path_to_append = Path(obs, image_obs, acs, rewards, next_obs, terminals)
    len_path_to_append = get_pathlength(path_to_append)
    with mp_timesteps_this_batch.get_lock():
      if mp_timesteps_this_batch.value >= min_timesteps_per_batch:
        enough_event.set()
      elif mp_timesteps_this_batch.value + len_path_to_append >= min_timesteps_per_batch:
        mp_paths.append(path_to_append)
        mp_timesteps_this_batch.value += len_path_to_append
        enough_event.set()
      else:
        mp_paths.append(path_to_append)
        mp_timesteps_this_batch.value += len_path_to_append

def sample_trajectory_NOCUDA(mp_timesteps_this_batch, min_timesteps_per_batch, mp_paths, enough_event,
                      env, policy, max_path_length, render=False, render_mode=('rgb_array')):
  while True:
    # initialize env for the beginning of a new rollout
    ob = env.reset()  # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
      # render image of the simulated env
      if render:
        if 'rgb_array' in render_mode:
          if hasattr(env, 'sim'):
            image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
          else:
            image_obs.append(env.render(mode=render_mode))
        if 'human' in render_mode:
          env.render(mode=render_mode)
          time.sleep(env.model.opt.timestep)

      # use the most recent ob to decide what to do
      obs.append(ob)

      ac = policy.get_action(ob)  # HINT: query the policy's get_action function
      ac = ac[0].cpu().detach().numpy()
      # ac = int(ac) # explicitly converted to int is only applicable to discrete action space
      acs.append(ac)
      # take that action and record results
      ob, rew, done, _ = env.step(ac)  # _ = info

      # record result of taking that action
      steps += 1
      next_obs.append(ob)
      rewards.append(rew)

      # HINT: rollout can end due to done, or due to max_path_length
      rollout_done = False  # HINT: this is either 0 or 1
      if done or steps >= max_path_length:
        rollout_done = True
      terminals.append(rollout_done)

      if rollout_done:
        break

    # obs lags one element to next_obs
    path_to_append = Path(obs, image_obs, acs, rewards, next_obs, terminals)
    len_path_to_append = get_pathlength(path_to_append)
    with mp_timesteps_this_batch.get_lock():
      if mp_timesteps_this_batch.value >= min_timesteps_per_batch:
        enough_event.set()
      elif mp_timesteps_this_batch.value + len_path_to_append >= min_timesteps_per_batch:
        mp_paths.append(path_to_append)
        mp_timesteps_this_batch.value += len_path_to_append
        enough_event.set()
      else:
        mp_paths.append(path_to_append)
        mp_timesteps_this_batch.value += len_path_to_append

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode='rgb_array'):
  """
      Collect rollouts until we have collected min_timesteps_per_batch steps.

      return paths, timesteps_this_batch

      Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
      Hint2: use get_pathlength to count the timesteps collected in each path
  """
  timesteps_this_batch = 0
  paths = []  # list of dict
  while timesteps_this_batch < min_timesteps_per_batch:
    path = sample_trajectory(env, policy, max_path_length, render, render_mode)
    paths.append(path)
    timesteps_this_batch += get_pathlength(path)

  return paths, timesteps_this_batch

def sample_trajectories_NOCUDA(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
  # *
  """
      Collect rollouts until we have collected min_timesteps_per_batch steps.

      return paths, timesteps_this_batch

      Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
      Hint2: use get_pathlength to count the timesteps collected in each path
  """
  import multiprocessing as mp
  try:
    mp.set_start_method('spawn')
  except RuntimeError:
    pass

  mp_paths_manager = mp.Manager()  # list of dict
  mp_paths = mp_paths_manager.list()  # list of dict
  mp_timesteps_this_batch = mp.Value('i', 0)

  # collect_traj = sample_trajectory(env, policy, max_path_length, render, render_mode)
  # results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
  enough_event = mp.Event()
  processes = []
  num_process = mp.cpu_count()
  num_process = 6  # spwan new thread also requires new gpu memory

  for p in range(num_process):
    p = mp.Process(target=sample_trajectory_NOCUDA,
                   args=(mp_timesteps_this_batch, min_timesteps_per_batch, mp_paths, enough_event,
                         env, policy, max_path_length, render, render_mode))
    p.start()
    processes.append(p)
  enough_event.wait()
  for p in processes:
    p.terminate()
  for p in processes:
    p.join()
  with mp_timesteps_this_batch.get_lock():
    mp_timesteps_this_batch = mp_timesteps_this_batch.value
  paths = list(mp_paths)
  return paths, mp_timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
  """
      Collect ntraj rollouts.

      Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
  """
  paths = []
  for i in range(ntraj):
    path = sample_trajectory(env, policy, max_path_length, render, render_mode)
    paths.append(path)

  return paths


############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
  """
      Take info (separate arrays) from a single rollout
      and return it in a single dictionary
  """
  if image_obs != []:
    image_obs = np.stack(image_obs, axis=0)
  return {"observation": np.array(obs, dtype=np.float32),
          "image_obs": np.array(image_obs, dtype=np.uint8),
          "reward": np.array(rewards, dtype=np.float32),
          "action": np.array(acs, dtype=np.float32),
          "next_observation": np.array(next_obs, dtype=np.float32),
          "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
  """
      Take a list of rollout dictionaries
      and return separate arrays,
      where each array is a concatenation of that array from across the rollouts
  """
  observations = np.concatenate([path["observation"] for path in paths])
  actions = np.concatenate([path["action"] for path in paths])
  next_observations = np.concatenate([path["next_observation"] for path in paths])
  terminals = np.concatenate([path["terminal"] for path in paths])
  concatenated_rewards = np.concatenate([path["reward"] for path in paths])
  unconcatenated_rewards = [path["reward"] for path in paths]
  return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards


############################################
############################################

def get_pathlength(path):
  return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
  return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
  return data * std + mean


def add_noise(data_inp, noiseToSignal=0.01):
  data = copy.deepcopy(data_inp)  # (num data points, dim)

  # mean of data
  mean_data = np.mean(data, axis=0)

  # if mean is 0,
  # make it 0.001 to avoid 0 issues later for dividing by std
  mean_data[mean_data == 0] = 0.000001

  # width of normal distribution to sample noise from
  # larger magnitude number = could have larger magnitude noise
  std_of_noise = mean_data * noiseToSignal
  for j in range(mean_data.shape[0]):
    data[:, j] = np.copy(data[:, j] + np.random.normal(
      0, np.absolute(std_of_noise[j]), (data.shape[0],)))

  return data
