from configs.linear_siren_scale10 import get_config as get_siren_config


def get_config():
  """Fixed-distance, single-direction squared-distance ablation."""
  config = get_siren_config()
  config.data.isometry_sampling_type = 'fixed_single'
  config.model.isometry_loss_type = 'squared_distance'
  return config
