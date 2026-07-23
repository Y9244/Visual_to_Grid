"""Training data generator for continuous two-dimensional movement."""

import ml_collections
import numpy as np


class TrainDataset:
  def __init__(
      self,
      config: ml_collections.ConfigDict,
      model_config: ml_collections.ConfigDict,
  ):
    self.config = config
    self.environment_size = model_config.environment_size
    if config.movement_distance_mode not in ('fixed', 'uniform'):
      raise ValueError(
          'movement_distance_mode must be "fixed" or "uniform", got '
          f'{config.movement_distance_mode!r}')
    if config.movement_dr <= 0:
      raise ValueError('movement_dr must be positive')
    if config.movement_dr_half_width < 0:
      raise ValueError('movement_dr_half_width must be non-negative')
    if (config.movement_distance_mode == 'uniform'
        and config.movement_dr <= config.movement_dr_half_width):
      raise ValueError(
          'movement_dr must be larger than movement_dr_half_width')

  def __iter__(self):
    while True:
      yield {'movement': self._gen_movement()}

  def _gen_movement(self):
    batch_size = self.config.batch_size
    theta = np.random.random(batch_size).astype(np.float32)
    theta *= np.float32(2 * np.pi)
    dr = self._sample_movement_distance(batch_size)
    dx = _dr_theta_to_dx(dr, theta)
    x = self._sample_valid_positions(dx)
    return {
        'x': x,
        'x_plus_dx': (x + dx).astype(np.float32),
    }

  def _sample_movement_distance(self, batch_size):
    if self.config.movement_distance_mode == 'fixed':
      return np.full(
          batch_size, self.config.movement_dr, dtype=np.float32)

    unit = np.random.random(batch_size).astype(np.float32)
    center = np.float32(self.config.movement_dr)
    half_width = np.float32(self.config.movement_dr_half_width)
    min_dr = center - half_width
    max_dr = center + half_width
    return (min_dr + (max_dr - min_dr) * unit).astype(np.float32)

  def _sample_valid_positions(self, dx):
    x_max = np.fmin(
        self.environment_size - 0.5,
        self.environment_size - 0.5 - dx)
    x_min = np.fmax(-0.5, -0.5 - dx)
    x = np.random.random(dx.shape).astype(np.float32)
    return (x * (x_max - x_min) + x_min).astype(np.float32)


def _dr_theta_to_dx(dr, theta):
  return np.stack(
      [dr * np.cos(theta), dr * np.sin(theta)], axis=-1).astype(np.float32)
