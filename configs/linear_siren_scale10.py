from configs.linear_scale10 import get_config as get_linear_config


def get_config():
  """Linear transformation model with a SIREN position encoder."""
  config = get_linear_config()
  config.train.positive_v = False
  config.model.encoder_type = 'siren'
  config.model.encoder_hidden_size = 128
  config.model.encoder_num_hidden_layers = 3
  config.model.siren_first_omega_0 = 30.0
  config.model.siren_hidden_omega_0 = 30.0
  config.model.encoder_softplus_beta = 1.0
  return config
