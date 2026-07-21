"""Create plots from a completed training run without TensorFlow/TensorBoard."""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _checkpoint_step(path):
  match = re.search(r'checkpoint-step(\d+)\.pth$', path.name)
  return int(match.group(1)) if match else -1


def _find_checkpoint(run_dir, step):
  checkpoint_dir = run_dir / 'ckpt'
  if step is not None:
    checkpoint = checkpoint_dir / f'checkpoint-step{step}.pth'
    if not checkpoint.exists():
      raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    return checkpoint

  checkpoints = sorted(checkpoint_dir.glob('checkpoint-step*.pth'),
                       key=_checkpoint_step)
  if not checkpoints:
    raise FileNotFoundError(f'No checkpoints found in {checkpoint_dir}')
  return checkpoints[-1]


def _load_checkpoint(path):
  return torch.load(path, map_location='cpu', weights_only=True)


def _load_metrics(path):
  with path.open(newline='') as metrics_file:
    rows = list(csv.DictReader(metrics_file))
  if not rows:
    raise ValueError(f'No metric rows found in {path}')
  return {key: np.asarray([float(row[key]) for row in rows])
          for key in rows[0]}


def plot_metrics(metrics_path, output_dir):
  metrics = _load_metrics(metrics_path)

  fig, axis = plt.subplots(figsize=(7, 4.5))
  for name in ('loss', 'loss_trans', 'loss_isometry'):
    axis.plot(metrics['step'], metrics[name], label=name)
  axis.set(xlabel='step', ylabel='loss', title='Training losses')
  axis.legend()
  axis.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(output_dir / 'losses.png', dpi=160)
  plt.close(fig)

  fig, axis = plt.subplots(figsize=(7, 4.5))
  for name in ('num_act', 'num_async'):
    axis.plot(metrics['step'], metrics[name], label=name)
  axis.set(xlabel='step', ylabel='count', title='Activity diagnostics')
  axis.legend()
  axis.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(output_dir / 'activity_metrics.png', dpi=160)
  plt.close(fig)


def plot_activity_table(activity, output_path):
  num_neurons = activity.shape[0]
  num_cols = min(8, num_neurons)
  num_rows = math.ceil(num_neurons / num_cols)
  fig, axes = plt.subplots(num_rows, num_cols,
                           figsize=(2.2 * num_cols, 2.2 * num_rows),
                           squeeze=False)
  for index, axis in enumerate(axes.flat):
    axis.axis('off')
    if index < num_neurons:
      image = axis.imshow(activity[index], cmap='jet', interpolation='nearest')
      axis.set_title(f'neuron {index}', fontsize=9)
      fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
  fig.suptitle('Grid-cell activity table (encoder.v)')
  fig.tight_layout()
  fig.savefig(output_path, dpi=160)
  plt.close(fig)


def plot_matrix_parameter(value, name, output_dir):
  """Plot A/B/b parameters for all model variants."""
  if value.ndim == 4:  # [theta, module, row, column]
    num_theta, num_modules = value.shape[:2]
    num_cols = min(6, num_theta)
    num_rows = math.ceil(num_theta / num_cols)
    for module in range(num_modules):
      fig, axes = plt.subplots(num_rows, num_cols,
                               figsize=(2.8 * num_cols, 2.8 * num_rows),
                               squeeze=False)
      limit = float(np.max(np.abs(value[:, module]))) or 1.0
      for theta, axis in enumerate(axes.flat):
        axis.axis('off')
        if theta < num_theta:
          image = axis.imshow(value[theta, module], cmap='coolwarm',
                              vmin=-limit, vmax=limit)
          axis.set_title(f'theta {theta}', fontsize=9)
          fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
      fig.suptitle(f'{name}, module {module}')
      fig.tight_layout()
      fig.savefig(output_dir / f'{name.replace(".", "_")}-module{module}.png',
                  dpi=160)
      plt.close(fig)
    return

  fig, axis = plt.subplots(figsize=(7, 6 if value.ndim == 2 else 4))
  if value.ndim == 2:
    limit = float(np.max(np.abs(value))) or 1.0
    image = axis.imshow(value, cmap='coolwarm', aspect='auto',
                        vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=axis)
    axis.set(xlabel='column / neuron', ylabel='row / theta')
  elif value.ndim == 1:
    axis.plot(value)
    axis.set(xlabel='index', ylabel='value')
    axis.grid(alpha=0.25)
  else:
    plt.close(fig)
    return
  axis.set_title(name)
  fig.tight_layout()
  fig.savefig(output_dir / f'{name.replace(".", "_")}.png', dpi=160)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('run_dir', type=Path,
                      help='Training run directory containing metrics.csv and ckpt/')
  parser.add_argument('--step', type=int, default=None,
                      help='Checkpoint step to visualize (default: latest)')
  parser.add_argument('--output-dir', type=Path, default=None)
  args = parser.parse_args()

  run_dir = args.run_dir.resolve()
  checkpoint_path = _find_checkpoint(run_dir, args.step)
  checkpoint = _load_checkpoint(checkpoint_path)
  step = int(checkpoint['step'])
  output_dir = (args.output_dir or
                (run_dir / 'visualizations' / f'step{step}')).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  plot_metrics(run_dir / 'metrics.csv', output_dir)
  state_dict = checkpoint['state_dict']
  activity = state_dict['encoder.v'].detach().cpu().numpy()
  plot_activity_table(activity, output_dir / 'encoder_v.png')

  for name, tensor in state_dict.items():
    if name == 'encoder.v':
      continue
    plot_matrix_parameter(tensor.detach().cpu().numpy(), name, output_dir)

  print(f'Visualizations written to {output_dir}')


if __name__ == '__main__':
  main()
