""" Main training loop. """
import csv
import json
import os
import time

from absl import logging
import ml_collections
import numpy as np
import torch

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
    ckpt_dir = os.path.join(workdir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(workdir, 'metrics.csv')
    metric_names = [
        'loss', 'loss_trans', 'loss_isometry', 'loss_norm',
        'num_act', 'num_async', 'module_norm_mean', 'module_norm_std',
    ]
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
        loss, metrics_step = self.model(batch_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()

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
              'step=%d/%d loss=%.6f loss_trans=%.6f '
              'loss_isometry=%.6f loss_norm=%.6f module_norm=%.6f',
              step, config.num_steps_train, row['loss'], row['loss_trans'],
              row['loss_isometry'], row['loss_norm'], row['module_norm_mean'])
          train_metrics = []

        if step % config.steps_per_large_logging == 0:
          self._save_checkpoint(step, ckpt_dir)

      if config.num_steps_train % config.steps_per_large_logging != 0:
        self._save_checkpoint(config.num_steps_train, ckpt_dir)

    logging.info('Training finished in %.1f seconds. Metrics: %s',
                 time.monotonic() - start_time, metrics_path)

  def _save_checkpoint(self, step, ckpt_dir):
    """Save tensor state and JSON metadata separately."""
    arch = type(self.model).__name__
    state = {
        'arch': arch,
        'step': step,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
    }
    filename = os.path.join(ckpt_dir, 'checkpoint-step{}.pth'.format(step))
    torch.save(state, filename)
    numpy_rng_state = np.random.get_state()
    metadata = {
        'config': self.config.to_dict(),
        'numpy_rng_state': {
            'bit_generator': numpy_rng_state[0],
            'state': numpy_rng_state[1].tolist(),
            'position': numpy_rng_state[2],
            'has_gauss': numpy_rng_state[3],
            'cached_gaussian': numpy_rng_state[4],
        },
    }
    metadata_filename = os.path.splitext(filename)[0] + '.json'
    with open(metadata_filename, 'w') as metadata_file:
      json.dump(metadata, metadata_file, indent=2)
    logging.info("Saving checkpoint: {} ...".format(filename))
