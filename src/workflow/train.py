import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ffcv.loader import Loader
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchattacks import APGD, CW, FAB, FGSM, PGD, DeepFool
from tqdm import tqdm

from schema.config import Config
from utils.dataset import get_dataloader


def train_clean_samples(
  config: Config,
  model: nn.Module,
  optimizer: Optimizer,
  criterion: nn.Module,
  train_dataloader: DataLoader[tuple[torch.Tensor, int]] | Loader,
):
  if config.resume.enable:
    assert config.checkpoint_path is not None
    checkpoints = list(config.checkpoint_path.glob('*.pt'))
    if config.checkpoint_path / 'last.pt' in checkpoints:
      checkpoints.remove(config.checkpoint_path / 'last.pt')

    old_steps = [int(pt.stem) for pt in checkpoints]

    steps = max(old_steps)
    if max(config.log_steps) <= steps:
      raise AttributeError('The resume step exceeds the max of config.log_steps.')

    checkpoint = torch.load(config.checkpoint_path / f'{steps}.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
  else:
    steps = 0

  log_steps = set(config.log_steps)
  logger = logging.getLogger('__main__')

  optimization_steps = config.train_adv_samples.interrupt_step \
    if config.train_adv_samples.enable \
    else config.train.optimization_steps

  with tqdm(total=optimization_steps) as pbar:
    pbar.update(steps)
    while steps < optimization_steps:
      for images, labels in train_dataloader:
        model.train()
        optimizer.zero_grad()

        if steps >= config.train.optimization_steps:
          break

        images = cast(torch.Tensor, images)
        labels = cast(torch.Tensor, labels)

        images = images.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        y_logit = model(images)
        loss = criterion(y_logit, labels)

        loss.backward()
        optimizer.step()

        if steps in log_steps:
          with torch.no_grad():
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()

          checkpoint: dict[str, Any] = {
            'step': steps,
            'model': model_state_dict,
            'optimizer': optimizer_state_dict
          }

          torch.save(checkpoint, config.output_path / f'{steps}.pt')

        pbar.update(1)
        steps += 1

def train_adv_samples(
  config: Config,
  resume_step: int,
  model: nn.Module,
  optimizer: Optimizer,
  criterion: nn.Module,
):
  raise NotImplementedError
  steps = resume_step
  log_steps = set(config.log_steps)

  adv_train_dataloader = get_dataloader(
    config,
    subset_size=config.train_adv_samples.train_size
  )

  with tqdm(total=config.train.optimization_steps) as pbar:
    pbar.update(steps)

    atk = PGD(model, config)
    while steps < config.train.optimization_steps:
      for images, labels in adv_train_dataloader:

        model.train()
        if steps >= config.train.optimization_steps:
          break

        images = cast(torch.Tensor, images)
        labels = cast(torch.Tensor, labels)

        images = images.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        images = atk(images, labels)

        optimizer.zero_grad()

        y_logit = model(images)
        loss = criterion(y_logit, labels)

        loss.backward()
        optimizer.step()

        if steps in log_steps:
          with torch.no_grad():
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()

          checkpoint: dict[str, Any] = {
            'step': steps,
            'model': model_state_dict,
            'optimizer': optimizer_state_dict
          }

          torch.save(checkpoint, config.output_path / f'{steps}.pt')

        pbar.update(1)
        steps += 1

def train(
  config: Config,
  model: nn.Module,
  optimizer: Optimizer,
  criterion: nn.Module,
  train_dataloader: DataLoader[tuple[torch.Tensor, int]] | Loader,
):
  train_clean_samples(
    config,
    model,
    optimizer,
    criterion,
    train_dataloader,
  )

  if config.train_adv_samples:
    train_adv_samples(
      config,
      config.train_adv_samples.interrupt_step,
      model,
      optimizer,
      criterion,
    )
