""" Data Generator """
import ml_collections
import numpy as np
# np.random.seed(0)

class TrainDataset:
  def __init__(self, config: ml_collections.ConfigDict, model_config: ml_collections.ConfigDict):
    self.config = config # config.data
    self.num_grid = model_config.num_grid # 40
    self.num_theta = model_config.num_theta # 18

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

    if self.num_theta is not None:
      theta_id = np.random.choice( # 0 - 17
          np.arange(self.num_theta), size=(batch_size,))
      theta = (theta_id * 2 * np.pi / self.num_theta).astype(np.float32)
    else:
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

    theta = np.random.random(size=(batch_size, 2)).astype(np.float32)
    theta *= np.float32(2 * np.pi)
    dr = np.sqrt(
        np.random.random(size=(batch_size, 1)).astype(np.float32))
    dr *= np.float32(config.max_dr_isometry)
    dx = _dr_theta_to_dx(dr, theta)  # [N, 2, 2]

    x_max = np.fmin(self.num_grid - 0.5, np.min(self.num_grid - 0.5 - dx, axis=1))
    x_min = np.fmax(-0.5, np.max(-0.5 - dx, axis=1))
    x = np.random.random(size=(batch_size, 2)).astype(np.float32)
    x = x * (x_max - x_min) + x_min
    x_plus_dx1 = x + dx[:, 0]
    x_plus_dx2 = x + dx[:, 1]

    return {
        'x': x.astype(np.float32),
        'x_plus_dx1': x_plus_dx1.astype(np.float32),
        'x_plus_dx2': x_plus_dx2.astype(np.float32),
    }


def _dr_theta_to_dx(dr, theta):
  dx_x = dr * np.cos(theta)
  dx_y = dr * np.sin(theta)
  dx = np.stack([dx_x, dx_y], axis=-1)

  return dx
