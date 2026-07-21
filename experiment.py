""" Main training loop. """
import csv
import os
import time

from absl import logging
import ml_collections
import numpy as np
import torch
import torch.nn as nn

import input_pipeline
import model as model
import utils

torch.manual_seed(1118)

class Experiment:
  def __init__(self, config: ml_collections.ConfigDict, device):
    self.config = config
    self.device = device

    # initialize models
    logging.info("==== initialize model ====")
    self.model_config = model.GridCellConfig(**config.model)
    self.model = model.GridCell(self.model_config).to(device)

    # initialize dataset
    logging.info("==== initialize dataset ====")
    train_dataset = input_pipeline.TrainDataset(config.data, self.model_config)
    self.train_iter = iter(train_dataset)

    # initialize optimizer
    logging.info("==== initialize optimizer ====")
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr)

  def train_and_evaluate(self, workdir):
    logging.info('==== Experiment.train_and_evaluate() ===')

    os.makedirs(workdir, exist_ok=True)
    config = self.config.train
    logging.info('num_steps_train=%d', config.num_steps_train)

    train_metrics = []
    module_size = self.model_config.module_size
    num_grid = self.model_config.num_grid
    num_module = self.model_config.num_neurons // module_size
    ckpt_dir = os.path.join(workdir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(workdir, 'metrics.csv')
    metric_names = ['loss', 'loss_trans', 'loss_isometry', 'num_act', 'num_async']
    start_time = time.monotonic()

    logging.info('==== Start of training ====')
    with open(metrics_path, 'w', newline='') as metrics_file:
      csv_writer = csv.DictWriter(
          metrics_file, fieldnames=['step', *metric_names])
      csv_writer.writeheader()

      for step in range(1, config.num_steps_train+1):
        batch_data = utils.dict_to_device(next(self.train_iter), self.device)

        # lr decay 
        decay_step = 6000 
        if step > decay_step: 
          lr = config.lr - (step - decay_step) * (config.lr / (config.num_steps_train - decay_step))
        elif step < 3000: 
          lr = config.lr * step / 3000
        else: 
          lr = config.lr
          
        for param_group in self.optimizer.param_groups: 
          param_group['lr'] = lr

        self.optimizer.zero_grad()
        loss, metrics_step = self.model(batch_data, step) # モデルのメイン部分
        loss.backward()
        torch.nn.utils.clip_grad_norm(parameters=self.model.parameters(), max_norm=10)
        self.optimizer.step()

        # Assumption 4. Non-negativity
        if config.positive_v:
          self.model.encoder.v.data = self.model.encoder.v.data.clamp(min=0.)

        # Assumption 3. Normalization
        if config.norm_v:
          with torch.no_grad():
            v = self.model.encoder.v.data.reshape(
                (-1, module_size, num_grid, num_grid))

            v_normed = nn.functional.normalize(v, dim=1) / np.sqrt(num_module)
            self.model.encoder.v.data = v_normed.reshape(
                (-1, num_grid, num_grid))
  
        metrics_step = utils.dict_to_numpy(metrics_step)
        train_metrics.append(metrics_step)

        # Quick indication that training is happening.
        logging.log_first_n(
            logging.WARNING, 'Finished training step %d.', 3, step)

        if (step % config.steps_per_logging == 0 or
            step == config.num_steps_train):
          train_metrics = utils.average_appended_metrics(train_metrics)
          row = {'step': step}
          row.update({name: float(train_metrics[name]) for name in metric_names})
          csv_writer.writerow(row)
          metrics_file.flush()
          logging.info(
              'step=%d/%d loss=%.6f loss_trans=%.6f loss_isometry=%.6f',
              step, config.num_steps_train, row['loss'], row['loss_trans'],
              row['loss_isometry'])
          train_metrics = []

        if step % config.steps_per_large_logging == 0:
          self._save_checkpoint(step, ckpt_dir)

      if config.num_steps_train % config.steps_per_large_logging != 0:
        self._save_checkpoint(config.num_steps_train, ckpt_dir)

    logging.info('Training finished in %.1f seconds. Metrics: %s',
                 time.monotonic() - start_time, metrics_path)

  def _save_checkpoint(self, step, ckpt_dir):
    """
    Saving checkpoints
    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(self.model).__name__
    state = {
        'arch': arch,
        'step': step,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'config': self.config,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
    }
    filename = os.path.join(ckpt_dir, 'checkpoint-step{}.pth'.format(step))
    torch.save(state, filename)
    logging.info("Saving checkpoint: {} ...".format(filename))

  def _resume_checkpoint(self, resume_path):
    """
    Resume from saved checkpoints
    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    self.logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, map_location=self.device)
    self.start_epoch = checkpoint['epoch'] + 1

    # load architecture params from checkpoint.
    if checkpoint['config']['arch'] != self.config['arch']:
      self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                          "checkpoint. This may yield an exception while state_dict is being loaded.")
    self.model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
      self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                          "Optimizer parameters not being resumed.")
    else:
      self.optimizer.load_state_dict(checkpoint['optimizer'])

    self.logger.info(
        "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
