import os

from absl import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import sys
import cv2
import imageio
import matplotlib.cm as cm
import math
import time
import torch
# if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

def draw_trajs(real_trajs, pred_trajs, area_size: int):
  # real_trajs: [N, T, 2], pred_trajs: [N, T, 2]
  nrow, ncol = 1, real_trajs.shape[0]
  assert real_trajs.shape == pred_trajs.shape
  
  fig = plt.figure(figsize=(ncol * 7, nrow * 7))

  for i in range(nrow):
    for j in range(ncol):
      plt.subplot(nrow, ncol, i * ncol + j + 1)
      _draw_real_pred_pairs(real_trajs[j], pred_trajs[j], area_size)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)

  return np.expand_dims(image_from_plot, axis=0)  


def _draw_real_pred_pairs(real, pred, area_size: int):
  # real: [N, 2], pred: [N, 2]
  plt.plot(real[:, 0], area_size - real[:, 1], c='k', lw=4, label='Real Path')
  plt.plot(pred[:, 0], area_size - pred[:, 1], 'o:', c='r', lw=4, ms=6, label='Inferred Path')
  plt.xlim((0, area_size))
  plt.ylim((0, area_size))
  plt.xticks([])
  plt.yticks([])
  plt.text(real[0, 0], area_size - real[0, 1] + 2, 'Start', fontsize=24, horizontalalignment='center', verticalalignment='center')
  plt.text(real[-1, 0], area_size - real[-1, 1] -2, 'End', fontsize=24, horizontalalignment='center', verticalalignment='center')
  plt.text(20, 3, '1.0 m', fontsize=24, horizontalalignment='center', verticalalignment='center')
  plt.annotate("", xy=(40, 3), xytext=(24, 3), arrowprops=dict(arrowstyle="->", lw=3))
  plt.annotate("", xy=(16, 3), xytext=(0, 3), arrowprops=dict(arrowstyle="<-", lw=3))

  plt.legend(loc='upper right', fontsize=18)
  ax = plt.gca()
  ax.set_aspect(1)


def draw_heatmap(weights):  
  # weights should a 4-D tensor: [M, N, H, W]
  # weights: [1, 10, 40, 40]
  nrow, ncol = weights.shape[0], weights.shape[1] # 1, 10
  fig = plt.figure(figsize=(ncol, nrow))

  for i in range(nrow):
    for j in range(ncol):
      plt.subplot(nrow, ncol, i * ncol + j + 1)
      weight = weights[i, j]
      vmin, vmax = weight.min() - 0.01, weight.max()

      cmap = cm.get_cmap('rainbow', 1000)
      cmap.set_under('w')

      plt.imshow(weight, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
      plt.axis('off')

  fig.canvas.draw()
  image, (width, height) = fig.canvas.print_to_buffer()
  image = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))[..., :3]
  plt.close(fig)

  return np.expand_dims(image, axis=0)


def save_heatmap(weights, name):  
  # weights should a 4-D tensor: [M, N, H, W]
  nrow, ncol = weights.shape[0], weights.shape[1]
  fig = plt.figure(figsize=(ncol, nrow))

  for i in range(nrow):
    for j in range(ncol):
      plt.subplot(nrow, ncol, i * ncol + j + 1)
      weight = weights[i, j]
      vmin, vmax = weight.min() - 0.01, weight.max()

      cmap = cm.get_cmap('rainbow', 1000)
      cmap.set_under('w')

      plt.imshow(weight, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
      plt.axis('off')

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.savefig(name)
  plt.close(fig)

  return np.expand_dims(image_from_plot, axis=0)


def average_appended_metrics(metrics):
  ks = metrics[0].keys()
  result = {k: np.mean([metrics[i][k]
                       for i in range(len(metrics))]) for k in ks}
  return result


def dict_to_numpy(data):
  for key, value in data.items():
    if isinstance(value, dict):
      data[key] = dict_to_numpy(data[key])
    else:
      data[key] = data[key].cpu().detach().numpy()

  return data


def dict_to_device(data, device):
  for key, value in data.items():
    if isinstance(value, dict):
      data[key] = dict_to_device(data[key], device)
    else:
      data[key] = torch.tensor(data[key]).to(device).float()
  
  return data


def _can_use_torch_device(device):
  try:
    torch.empty(1, device=device)
    return True
  except Exception:
    return False


def get_device(device=0):
  if torch.cuda.is_available():
    cuda_device = torch.device('cuda:{}'.format(device))
    if _can_use_torch_device(cuda_device):
      return cuda_device

  if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    mps_device = torch.device('mps')
    if _can_use_torch_device(mps_device):
      return mps_device

  return torch.device('cpu')


def set_gpu(gpu, deterministic=True):
  if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
    if not deterministic:
      torch.backends.cudnn.benchmark = True
      torch.backends.cudnn.deterministic = False
    else:
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True


""" Run with temporary verbosity """


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
  argv = sys.argv
  config_list = []
  config_list.append(time.strftime('%Y%m%d-%H%M%S'))
  for i in range(1, len(argv)):
    if argv[i].startswith('--config='):
      config_file = argv[i].split('/')[-1]
      config_file = config_file.split('.py')[0]
    elif argv[i].startswith('--workdir='):
      continue
    else:
      cfg = argv[i].split('.')[-1]
      if cfg == '0':
        cfg = argv[i].split('.')[-2]
      config_list.append(cfg)
  workdir = "-".join(config_list)

  return os.path.join(config_file, workdir)


def shape_mask(size, shape):
  x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
  if shape == 'square':
    mask = np.ones_like(x, dtype=bool)
  elif shape == 'circle':
    mask = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) <= 0.5
  elif shape == 'triangle':
    mask = (y + 2 * x >= 1) * (-y + 2 * x <= 1)
  else:
    raise NotImplementedError

  return mask


def draw_heatmap_2D(data, vmin=None, vmax=None, shape='square', cb=False):
  place_size, _ = np.shape(data)
  place_mask = shape_mask(place_size, shape)
  if vmin is None:
    vmin = data[place_mask].min()
  if vmax is None:
    vmax = data[place_mask].max()
  # data[~place_mask] = vmin - 1

  cmap = cm.get_cmap('rainbow', 1000)
  cmap.set_under('w')
  plt.imshow(data, interpolation='nearest', cmap=cmap,
             aspect='auto', vmin=vmin, vmax=vmax)
  if cb:
    plt.colorbar()
  plt.axis('off')


def draw_path_to_target(place_len, place_seq, save_file=None, target=None, obstacle=None, a=None, b=None, col_scheme='single'):
  if save_file is not None:
    plt.figure(figsize=(5, 5))
  if type(place_seq) == list or np.ndim(place_seq) > 2:
    if col_scheme == 'rainbow':
      colors = cm.rainbow(np.linspace(0, 1, len(place_seq)))
    for i in range(len(place_seq)):
      p_seq = place_seq[i]
      color = colors[i] if col_scheme == 'rainbow' else 'darkcyan'
      label = 'a=%.2f, b=%d' % (a[i], b[i]) if (
          a is not None and b is not None) else ''
      plt.plot(p_seq[:, 1], place_len - p_seq[:, 0], 'o-', color=color,
               ms=6, lw=2.0, label=label)
      if a is not None and len(a) > 1:
        plt.legend(loc='lower left', fontsize=12)
  else:
    if type(place_seq) == list:
      place_seq = place_seq[0]
    plt.plot(place_seq[:, 1], place_len - place_seq[:, 0],
             'o-', ms=6, lw=2.0, color='darkcyan')

  if target is not None:
    plt.plot(target[1], place_len - target[0], '*', ms=12, color='r')
    if obstacle is not None:
      if np.ndim(obstacle) == 2:
        obstacle = obstacle.T
      plt.plot(obstacle[1], place_len - obstacle[0],
               's', ms=8, color='dimgray')

  plt.xticks([])
  plt.yticks([])
  plt.xlim((0, place_len))
  plt.ylim((0, place_len))
  ax = plt.gca()
  ax.set_aspect(1)
  if save_file is not None:
    plt.savefig(save_file)
    plt.close()


def draw_path_to_target_planning(place_len, place_seq, scale, save_file=None, target=None, obstacle=None, a=None, b=None, col_scheme='single'):
  plt.figure(figsize=(5, 5))
  ax = plt.gca()

  def plot_by_scale(s, label):
    color = next(ax._get_lines.prop_cycler)['color']
    idx = np.where(scale == s)[0] + 1
    plt.scatter(place_seq[idx, 1], place_len -
                place_seq[idx, 0], label=label, color=color)
    for id in idx:
      plt.plot(place_seq[(id-1, id), 1], place_len -
               place_seq[(id-1, id), 0], color=color)
  plot_by_scale(0, "$\sigma = 0.07$")
  plot_by_scale(1, "$\sigma = 0.14$")
  plot_by_scale(2, "$\sigma = 0.28$")
  plt.legend(loc='best', fontsize=14)

  if target is not None:
    plt.plot(target[1], place_len - target[0], '*', ms=12, color='r')
    if obstacle is not None:
      if np.ndim(obstacle) == 2:
        obstacle = obstacle.T
      plt.plot(obstacle[1], place_len - obstacle[0],
               's', ms=8, color='dimgray')

  plt.xticks([])
  plt.yticks([])
  plt.xlim((30, 50))
  plt.ylim((30, 50))
  ax = plt.gca()
  ax.set_aspect(1)
  if save_file is not None:
    plt.savefig(save_file)
    plt.close()


def draw_two_path(place_len, place_gt, place_pd):
  place_gt = np.round(place_gt).astype(int)
  place_pd = np.round(place_pd).astype(int)
  plt.plot(place_gt[:, 0], place_len - place_gt[:, 1],
           c='k', lw=4, label='Real Path')
  plt.plot(place_pd[:, 0], place_len - place_pd[:, 1],
           'o:', c='r', lw=4, ms=6, label='Inferred Path')
  plt.xlim((0, place_len))
  plt.ylim((0, place_len))
  plt.xticks([])
  plt.yticks([])
  plt.text(place_gt[0, 0], place_len - place_gt[0, 1] + 2, 'Start',
           fontsize=24, horizontalalignment='center', verticalalignment='center')
  plt.text(place_gt[-1, 0], place_len - place_gt[-1, 1] - 2, 'End',
           fontsize=24, horizontalalignment='center', verticalalignment='center')
  plt.text(20, 3, '1.0 m', fontsize=24,
           horizontalalignment='center', verticalalignment='center')
  plt.annotate("", xy=(40, 3), xytext=(24, 3),
               arrowprops=dict(arrowstyle="->", lw=3))
  plt.annotate("", xy=(16, 3), xytext=(0, 3),
               arrowprops=dict(arrowstyle="<-", lw=3))

  plt.legend(loc='upper right', fontsize=18)
  ax = plt.gca()
  ax.set_aspect(1)


def draw_path_integral(place_len, place_seq, col=(255, 0, 0)):
  place_seq = np.round(place_seq).astype(int)
  cmap = cm.get_cmap('rainbow', 1000)

  canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
  if target is not None:
    cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
    cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)
  else:
    cv2.circle(canvas, tuple(place_seq[-1]), 2, col, -1)
  for i in range(len(place_seq) - 1):
    cv2.line(canvas, tuple(place_seq[i]), tuple(place_seq[i+1]), col, 1)

  plt.imshow(np.swapaxes(canvas, 0, 1),
             interpolation='nearest', cmap=cmap, aspect='auto')
  return canvas


def draw_path_to_target_gif(file_name, place_len, place_seq, target, col=(255, 0, 0)):
  cmap = cm.get_cmap('rainbow', 1000)
  canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
  cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
  cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)

  canvas_list = []
  canvas_list.append(canvas)
  for i in range(1, len(place_seq)):
    canvas = np.ones((place_len, place_len, 3), dtype="uint8") * 255
    cv2.circle(canvas, tuple(target), 2, (0, 0, 255), -1)
    cv2.circle(canvas, tuple(place_seq[0]), 2, col, -1)
    for j in range(i):
      cv2.line(canvas, tuple(place_seq[j]), tuple(place_seq[j+1]), col, 1)
    canvas_list.append(canvas)

  imageio.mimsave(file_name, canvas_list, 'GIF', duration=0.3)


def mu_to_map_old(mu, num_interval, max=1.0):
  if len(mu.shape) == 1:
    map = np.zeros([num_interval, num_interval], dtype=np.float32)
    map[mu[0], mu[1]] = max
  elif len(mu.shape) == 2:
    map = np.zeros([mu.shape[0], num_interval, num_interval], dtype=np.float32)
    for i in range(len(mu)):
      map[i, mu[i, 0], mu[i, 1]] = 1.0

  return map


def mu_to_map(mu, num_interval):
  mu = mu / float(num_interval)
  if len(mu.shape) == 1:
    discretized_x = np.expand_dims(np.linspace(0, 1, num=num_interval), axis=1)
    max_pdf = pow(norm.pdf(0, loc=0, scale=0.02), 2)
    vec_x_before = norm.pdf(discretized_x, loc=mu[0], scale=0.02)
    vec_y_before = norm.pdf(discretized_x, loc=mu[1], scale=0.02)
    map = np.dot(vec_x_before, vec_y_before.T) / max_pdf
  elif len(mu.shape) == 2:
    map_list = []
    max_pdf = pow(norm.pdf(0, loc=0, scale=0.005), 2)
    for i in range(len(mu)):
      discretized_x = np.expand_dims(
          np.linspace(0, 1, num=num_interval), axis=1)
      vec_x_before = norm.pdf(discretized_x, loc=mu[i, 0], scale=0.005)
      vec_y_before = norm.pdf(discretized_x, loc=mu[i, 1], scale=0.005)
      map = np.dot(vec_x_before, vec_y_before.T) / max_pdf
      map_list.append(map)
    map = np.stack(map_list, axis=0)

  return map


def get_argv():
  argv = sys.argv
  for i in range(1, len(argv)):
    if argv[i] == '--ckpt':
      argv.pop(i)
      argv.pop(i)
      break
  for i in range(1, len(argv)):
    if argv[i].startswith('--ckpt='):
      argv.pop(i)
      break
  for i in range(1, len(argv)):
    if argv[i] == '--gpu':
      argv.pop(i)
      argv.pop(i)
      break
  for i in range(1, len(argv)):
    if argv[i].startswith('--gpu='):
      argv.pop(i)
      break
  return ''.join(argv[1:])
