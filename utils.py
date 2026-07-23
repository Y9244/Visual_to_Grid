import os
import sys
import time

from absl import logging
import numpy as np
from scipy.stats import norm
import torch


def set_random_seed(seed):
  """Seed training-data sampling and model initialization."""
  np.random.seed(seed)
  torch.manual_seed(seed)


def average_appended_metrics(metrics):
  keys = metrics[0].keys()
  return {
      key: np.mean([metric[key] for metric in metrics])
      for key in keys
  }


def dict_to_numpy(data):
  for key, value in data.items():
    if isinstance(value, dict):
      data[key] = dict_to_numpy(value)
    else:
      data[key] = value.detach().cpu().numpy()
  return data


def dict_to_device(data, device):
  for key, value in data.items():
    if isinstance(value, dict):
      data[key] = dict_to_device(value, device)
    else:
      data[key] = torch.as_tensor(
          value, dtype=torch.float32, device=device)
  return data


def _can_use_torch_device(device):
  try:
    torch.empty(1, device=device)
    return True
  except Exception:
    return False


def get_device(device=0):
  if torch.cuda.is_available():
    cuda_device = torch.device(f'cuda:{device}')
    if _can_use_torch_device(cuda_device):
      return cuda_device

  if torch.backends.mps.is_available():
    mps_device = torch.device('mps')
    if _can_use_torch_device(mps_device):
      return mps_device

  return torch.device('cpu')


def set_gpu(gpu, deterministic=True):
  if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def with_verbosity(temporary_verbosity_level, fn):
  old_verbosity_level = logging.get_verbosity()
  logging.set_verbosity(temporary_verbosity_level)
  result = fn()
  logging.set_verbosity(old_verbosity_level)
  return result


def copy_source(file, output_dir):
  import shutil
  shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_workdir():
  config_list = [time.strftime('%Y%m%d-%H%M%S')]
  for argument in sys.argv[1:]:
    if argument.startswith('--config='):
      config_file = argument.split('/')[-1].split('.py')[0]
    elif argument.startswith('--workdir='):
      continue
    else:
      config_value = argument.split('.')[-1]
      if config_value == '0':
        config_value = argument.split('.')[-2]
      config_list.append(config_value)
  return os.path.join(config_file, '-'.join(config_list))


def mu_to_map_old(mu, num_interval, max=1.0):
  if len(mu.shape) == 1:
    result = np.zeros([num_interval, num_interval], dtype=np.float32)
    result[mu[0], mu[1]] = max
  elif len(mu.shape) == 2:
    result = np.zeros(
        [mu.shape[0], num_interval, num_interval], dtype=np.float32)
    for index in range(len(mu)):
      result[index, mu[index, 0], mu[index, 1]] = 1.0
  return result


def mu_to_map(mu, num_interval):
  mu = mu / float(num_interval)
  if len(mu.shape) == 1:
    discretized_x = np.expand_dims(
        np.linspace(0, 1, num=num_interval), axis=1)
    max_pdf = pow(norm.pdf(0, loc=0, scale=0.02), 2)
    vec_x = norm.pdf(discretized_x, loc=mu[0], scale=0.02)
    vec_y = norm.pdf(discretized_x, loc=mu[1], scale=0.02)
    return np.dot(vec_x, vec_y.T) / max_pdf

  maps = []
  max_pdf = pow(norm.pdf(0, loc=0, scale=0.005), 2)
  for value in mu:
    discretized_x = np.expand_dims(
        np.linspace(0, 1, num=num_interval), axis=1)
    vec_x = norm.pdf(discretized_x, loc=value[0], scale=0.005)
    vec_y = norm.pdf(discretized_x, loc=value[1], scale=0.005)
    maps.append(np.dot(vec_x, vec_y.T) / max_pdf)
  return np.stack(maps, axis=0)


def get_argv():
  argv = sys.argv
  for name in ('--ckpt', '--gpu'):
    for index in range(1, len(argv)):
      if argv[index] == name:
        argv.pop(index)
        argv.pop(index)
        break
      if argv[index].startswith(f'{name}='):
        argv.pop(index)
        break
  return ''.join(argv[1:])
