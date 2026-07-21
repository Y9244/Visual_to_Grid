"""Measure hexagonal and square gridness from a trained encoder."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal

from visualize_results import (
    _find_checkpoint,
    _load_checkpoint,
    reconstruct_encoder_activity,
)


ROTATION_ANGLES = (30, 45, 60, 90, 120, 135, 150)


def spatial_autocorrelation(rate_map):
  centered = rate_map.astype(np.float64) - np.mean(rate_map)
  correlation = signal.fftconvolve(
      centered, centered[::-1, ::-1], mode='full')
  overlap = signal.fftconvolve(
      np.ones_like(centered), np.ones_like(centered), mode='full')
  autocorrelation = correlation / np.maximum(overlap, 1.0)
  variance = np.mean(centered ** 2)
  return autocorrelation / max(variance, np.finfo(np.float64).eps)


def first_ring_radius(autocorrelation):
  """Estimate the radius of the six nearest non-central correlation peaks."""
  smoothed = ndimage.gaussian_filter(autocorrelation, sigma=1.5)
  height, width = smoothed.shape
  center_y, center_x = (height - 1) / 2, (width - 1) / 2
  y, x = np.indices(smoothed.shape)
  radius = np.hypot(x - center_x, y - center_y)
  max_radius = min(height, width) * 0.24

  local_maximum = smoothed == ndimage.maximum_filter(smoothed, size=9)
  candidates = np.argwhere(
      local_maximum & (radius >= 6) & (radius <= max_radius) &
      (smoothed >= 0.1))
  if len(candidates) >= 6:
    distances = np.hypot(
        candidates[:, 1] - center_x, candidates[:, 0] - center_y)
    values = smoothed[candidates[:, 0], candidates[:, 1]]
    # Prefer strong peaks, then use the nearest six among that reliable set.
    reliable = np.argsort(values)[-min(len(values), 18):]
    nearest = np.sort(distances[reliable])[:6]
    return float(np.median(nearest))

  # Conservative fallback for maps whose first ring is not cleanly separated.
  return min(height, width) * 0.16


def rotation_correlation(autocorrelation, angle, mask):
  rotated = ndimage.rotate(
      autocorrelation, angle, reshape=False, order=1,
      mode='constant', cval=np.nan)
  valid = mask & np.isfinite(rotated)
  original_values = autocorrelation[valid]
  rotated_values = rotated[valid]
  if original_values.size < 3:
    return float('nan')
  return float(np.corrcoef(original_values, rotated_values)[0, 1])


def measure_gridness(rate_map):
  autocorrelation = spatial_autocorrelation(rate_map)
  ring_radius = first_ring_radius(autocorrelation)
  height, width = autocorrelation.shape
  center_y, center_x = (height - 1) / 2, (width - 1) / 2
  y, x = np.indices(autocorrelation.shape)
  radius = np.hypot(x - center_x, y - center_y)
  inner_radius = max(4.0, ring_radius * 0.5)
  outer_radius = min(min(height, width) * 0.24, ring_radius * 1.5)
  mask = (radius >= inner_radius) & (radius <= outer_radius)

  correlations = {
      angle: rotation_correlation(autocorrelation, angle, mask)
      for angle in ROTATION_ANGLES
  }
  hex_gridness = min(correlations[60], correlations[120]) - max(
      correlations[30], correlations[90], correlations[150])
  square_gridness = correlations[90] - max(
      correlations[45], correlations[135])
  return {
      'autocorrelation': autocorrelation,
      'ring_radius': ring_radius,
      'inner_radius': inner_radius,
      'outer_radius': outer_radius,
      'hex_gridness': hex_gridness,
      'square_gridness': square_gridness,
      'correlations': correlations,
  }


def plot_autocorrelations(results, output_path):
  num_neurons = len(results)
  num_cols = min(8, num_neurons)
  num_rows = math.ceil(num_neurons / num_cols)
  fig, axes = plt.subplots(
      num_rows, num_cols, figsize=(2.4 * num_cols, 2.4 * num_rows),
      squeeze=False)
  for neuron, axis in enumerate(axes.flat):
    axis.axis('off')
    if neuron >= num_neurons:
      continue
    result = results[neuron]
    autocorrelation = result['autocorrelation']
    limit = np.nanmax(np.abs(autocorrelation)) or 1.0
    axis.imshow(
        autocorrelation, cmap='coolwarm', vmin=-limit, vmax=limit)
    center = (np.asarray(autocorrelation.shape) - 1) / 2
    for radius in (result['inner_radius'], result['outer_radius']):
      axis.add_patch(plt.Circle(
          (center[1], center[0]), radius, fill=False,
          color='black', linewidth=0.6))
    axis.set_title(
        f'n{neuron}: hex={result["hex_gridness"]:.2f}\n'
        f'square={result["square_gridness"]:.2f}', fontsize=8)
  fig.suptitle('Spatial autocorrelations and gridness annuli')
  fig.tight_layout()
  fig.savefig(output_path, dpi=160)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('run_dir', type=Path)
  parser.add_argument('--step', type=int, default=None)
  parser.add_argument('--output-dir', type=Path, default=None)
  args = parser.parse_args()

  run_dir = args.run_dir.resolve()
  checkpoint_path = _find_checkpoint(run_dir, args.step)
  checkpoint = _load_checkpoint(checkpoint_path)
  activity = reconstruct_encoder_activity(
      checkpoint_path, checkpoint['state_dict'])
  step = int(checkpoint['step'])
  output_dir = (args.output_dir or
                run_dir / 'gridness' / f'step{step}').resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  results = [measure_gridness(rate_map) for rate_map in activity]
  fieldnames = [
      'neuron', 'hex_gridness', 'square_gridness', 'ring_radius',
      *[f'rotation_{angle}' for angle in ROTATION_ANGLES],
  ]
  with (output_dir / 'gridness.csv').open('w', newline='') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    for neuron, result in enumerate(results):
      writer.writerow({
          'neuron': neuron,
          'hex_gridness': result['hex_gridness'],
          'square_gridness': result['square_gridness'],
          'ring_radius': result['ring_radius'],
          **{
              f'rotation_{angle}': result['correlations'][angle]
              for angle in ROTATION_ANGLES
          },
      })

  plot_autocorrelations(results, output_dir / 'autocorrelations.png')
  hex_scores = np.asarray([result['hex_gridness'] for result in results])
  square_scores = np.asarray([
      result['square_gridness'] for result in results])
  print(f'Gridness written to {output_dir}')
  print(
      f'hex: mean={hex_scores.mean():.3f}, median={np.median(hex_scores):.3f}, '
      f'positive={np.sum(hex_scores > 0)}/{len(hex_scores)}')
  print(
      f'square: mean={square_scores.mean():.3f}, '
      f'median={np.median(square_scores):.3f}')
  print(
      f'hex > square: {np.sum(hex_scores > square_scores)}/'
      f'{len(hex_scores)}')


if __name__ == '__main__':
  main()
