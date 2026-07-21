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
    self.num_grid = model_config.num_grid

  def __iter__(self):
    while True:
      yield {
          'trans': self._gen_data_trans(),
          'isometry': self._gen_data_isometry(),
      }

  def _gen_data_trans(self):
    batch_size = self.config.batch_size
    theta = np.random.random(batch_size).astype(np.float32)
    theta *= np.float32(2 * np.pi)
    dr = np.sqrt(np.random.random(batch_size).astype(np.float32))
    dr *= np.float32(self.config.max_dr_trans)
    dx = _dr_theta_to_dx(dr, theta)
    x = self._sample_valid_positions(dx)
    return {
        'x': x,
        'x_plus_dx': (x + dx).astype(np.float32),
    }

  def _gen_data_isometry(self):
    batch_size = self.config.batch_size
    theta = np.random.random(batch_size).astype(np.float32)
    theta *= np.float32(2 * np.pi)
    dr = np.full(
        batch_size, self.config.fixed_dr_isometry, dtype=np.float32)
    dx = _dr_theta_to_dx(dr, theta)
    x = self._sample_valid_positions(dx)
    return {
        'x': x,
        'x_plus_dx': (x + dx).astype(np.float32),
    }

  def _sample_valid_positions(self, dx):
    x_max = np.fmin(
        self.num_grid - 0.5, self.num_grid - 0.5 - dx)
    x_min = np.fmax(-0.5, -0.5 - dx)
    x = np.random.random(dx.shape).astype(np.float32)
    return (x * (x_max - x_min) + x_min).astype(np.float32)


def _dr_theta_to_dx(dr, theta):
  return np.stack(
      [dr * np.cos(theta), dr * np.sin(theta)], axis=-1).astype(np.float32)
