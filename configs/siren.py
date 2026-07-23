import ml_collections


def _config_dict(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """The supported SIREN grid-cell training configuration."""
  config = ml_collections.ConfigDict()
  config.gpu = 0
  config.seed = 0
  config.train = _config_dict(
      num_steps_train=20000,
      lr=0.003,
      min_lr=0.0001,
      warmup_steps=500,
      decay_start_step=12000,
      steps_per_logging=20,
      steps_per_large_logging=500,
  )
  config.data = _config_dict(
      movement_distance_mode='fixed',  # uniform, fixed
      movement_dr=5.5,
      movement_dr_half_width=0.5,
      target_activity_distance=0.8,
      batch_size=4000,
  )
  config.model = _config_dict(
      environment_size=40,
      num_neurons=24,
      module_size=24,
      w_trans=1.0,
      w_isometry=32.0,
      w_norm=1.0,
      trans_hidden_size=128,
      trans_num_hidden_layers=2,
      encoder_hidden_size=128,
      encoder_num_hidden_layers=3,
      siren_first_omega_0=30.0,
      siren_hidden_omega_0=30.0,
      encoder_softplus_beta=1.0,
  )
  return config
