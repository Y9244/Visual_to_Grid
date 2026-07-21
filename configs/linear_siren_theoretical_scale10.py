from configs.linear_siren_scale10 import get_config as get_siren_config


def get_config():
  """SIREN experiment using the paper's squared-distance objective."""
  config = get_siren_config()
  config.data.isometry_sampling_type = 'fixed_multi'
  config.data.fixed_dr_isometry = True
  config.data.num_isometry_directions = 12
  config.model.isometry_loss_type = 'squared_distance'
  return config
