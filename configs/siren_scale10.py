import ml_collections


def _config_dict(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """The supported SIREN grid-cell training configuration."""
  config = ml_collections.ConfigDict()
  config.gpu = 0
  config.train = _config_dict(
      num_steps_train=10000,
      lr=0.003,
      steps_per_logging=20,
      steps_per_large_logging=500,
  )
  config.data = _config_dict(
      max_dr_trans=3.0,
      fixed_dr_isometry=5.0,
      batch_size=4000,
  )
  config.model = _config_dict(
      num_grid=40,
      num_neurons=24,
      module_size=24,
      w_trans=1.0,
      w_isometry=10.0,
      w_norm=1.0,
      s_fixed=10.0,
      trans_hidden_size=128,
      trans_num_hidden_layers=2,
      encoder_hidden_size=128,
      encoder_num_hidden_layers=3,
      siren_first_omega_0=30.0,
      siren_hidden_omega_0=30.0,
      encoder_softplus_beta=1.0,
  )
  return config
