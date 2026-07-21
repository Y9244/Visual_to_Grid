""" Data Generator """
import ml_collections
import numpy as np
# np.random.seed(0)

class TrainDataset:
  def __init__(self, config: ml_collections.ConfigDict, model_config: ml_collections.ConfigDict):
    self.config = config # config.data
    self.num_grid = model_config.num_grid # 40
    self.isometry_loss_type = model_config.isometry_loss_type

    self.num_blocks = model_config.num_neurons // model_config.module_size # 1
    self.scale_vector = (
        np.zeros(self.num_blocks, dtype=np.float32) + config.max_dr_isometry)

  def __iter__(self):
    while True:
      yield {
          "trans": self._gen_data_trans(),            # 移動後の表現を予測するloss
          "isometry": self._gen_data_iso_numerical(), # 距離保存loss
      }

  def _gen_data_trans(self):
    """
    {'x': x, 'x_plus_dx': x_plus_dx}を生成するメソッド
    x, x_plus_dxの両方とも[batch_size, 2]
    移動前と移動後のベクトルが入っている

    各位置の範囲は[-0.5, num_grid-0.5]に収まるように設定されている。
    num_gridはグリッド細胞数ではなく、環境の格子数=40。つまり、[-0.5, 39.5]
    """
    batch_size = self.config.batch_size
    config = self.config

    theta = (
        np.random.random(size=(batch_size,)).astype(np.float32) *
        np.float32(2 * np.pi))
    
    dr = np.sqrt(
        np.random.random(size=(batch_size,)).astype(np.float32))
    dr *= np.float32(config.max_dr_trans)
    dx = _dr_theta_to_dx(dr, theta) # [batch_size, 2]

    x_max = np.fmin(self.num_grid - 0.5, self.num_grid - 0.5 - dx)
    x_min = np.fmax(-0.5, -0.5 - dx)
    x = np.random.random(size=(batch_size, 2)).astype(np.float32)
    x = x * (x_max - x_min) + x_min
    x_plus_dx = x + dx

    # x, x_plus_dx: [batch_size, 2], [batch_size, 2]
    return {
        'x': x.astype(np.float32),
        'x_plus_dx': x_plus_dx.astype(np.float32),
    }

  def _gen_data_iso_numerical(self):
    batch_size = self.config.batch_size
    config = self.config

    sampling_type = getattr(
        config, 'isometry_sampling_type', 'random_single')
    if sampling_type in ('random_single', 'fixed_single'):
      theta = np.random.random(size=(batch_size, 2)).astype(np.float32)
      theta *= np.float32(2 * np.pi)
      if sampling_type == 'fixed_single':
        dr = np.full(
            (batch_size, 1), config.max_dr_isometry, dtype=np.float32)
      else:
        dr = np.sqrt(
            np.random.random(size=(batch_size, 1)).astype(np.float32))
        dr *= np.float32(config.max_dr_isometry)
      dx = _dr_theta_to_dx(dr, theta)
      x_max = np.fmin(
          self.num_grid - 0.5,
          np.min(self.num_grid - 0.5 - dx, axis=1))
      x_min = np.fmax(-0.5, np.max(-0.5 - dx, axis=1))
      x = np.random.random(size=(batch_size, 2)).astype(np.float32)
      x = x * (x_max - x_min) + x_min
      return {
          'x': x.astype(np.float32),
          'x_plus_dx1': (x + dx[:, 0]).astype(np.float32),
          'x_plus_dx2': (x + dx[:, 1]).astype(np.float32),
      }

    if sampling_type != 'fixed_multi':
      raise ValueError(
          f'Unknown isometry_sampling_type: {sampling_type}')

    num_directions = getattr(config, 'num_isometry_directions', 12)
    theta_offset = np.random.random(
        size=(batch_size, 1)).astype(np.float32)
    theta_offset *= np.float32(2 * np.pi / num_directions)
    theta = np.arange(num_directions, dtype=np.float32)[None, :]
    theta *= np.float32(2 * np.pi / num_directions)
    theta = theta_offset + theta

    if getattr(config, 'fixed_dr_isometry', True):
      dr = np.full(
          (batch_size, 1), config.max_dr_isometry, dtype=np.float32)
    else:
      dr = np.sqrt(
          np.random.random(size=(batch_size, 1)).astype(np.float32))
      dr *= np.float32(config.max_dr_isometry)
    dx = _dr_theta_to_dx(dr, theta)  # [N, num_directions, 2]

    x_max = np.fmin(self.num_grid - 0.5, np.min(self.num_grid - 0.5 - dx, axis=1))
    x_min = np.fmax(-0.5, np.max(-0.5 - dx, axis=1))
    x = np.random.random(size=(batch_size, 2)).astype(np.float32)
    x = x * (x_max - x_min) + x_min
    x_plus_dx = x[:, None, :] + dx

    return {
        'x': x.astype(np.float32),
        'x_plus_dx': x_plus_dx.astype(np.float32),
    }


def _dr_theta_to_dx(dr, theta):
  dx_x = dr * np.cos(theta)
  dx_y = dr * np.sin(theta)
  dx = np.stack([dx_x, dx_y], axis=-1)

  return dx
