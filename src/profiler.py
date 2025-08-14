import argparse
import logging
import logging.config
import os
import random
from pathlib import Path
from typing import Any, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import wandb
import yaml
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from attacks import PGD
from schema.config import Config
from utils.constants import (
  get_dataset,
  get_loss_func,
  get_model,
  get_optimizer,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs'

parser = argparse.ArgumentParser()

parser.add_argument('--config', '-c', help='config file', default='main.yaml')
args = parser.parse_args()

scaler = torch.GradScaler()

def seed_everything(seed: int=0):
  """Fix all random seeds"""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
  os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id: int):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)
  # worker_info = torch.utils.data.get_worker_info()
  # dataset = worker_info.dataset
  # dataset._init_file()

g = torch.Generator()
g.manual_seed(0)

def train_one_epoch(
  config: Config,
  model: nn.Module,
  optimizer: Optimizer,
  criterion: nn.Module,
  dataloader: DataLoader[tuple[torch.Tensor, int]],
):
  model.train()

  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  # one_hots = torch.eye(10, 10, dtype=torch.float32, device=config.device)
  with torch.profiler.profile(
      activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
      ],
      record_shapes=True,
      profile_memory=True,
      with_stack=True
  ) as prof:

    for images, labels in dataloader:
      optimizer.zero_grad()

      images = images.to(config.device).to(memory_format=torch.channels_last)
      labels = labels.to(config.device)
      with torch.autocast(config.device, dtype=torch.float16):
        y_logit = model(images).squeeze(1)
        loss = criterion(y_logit, labels)
      total_loss += loss.item()

      # loss.backward()
      # optimizer.step()

      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_norm)
      scaler.step(optimizer)
      scaler.update()

      y_prob = F.softmax(y_logit, dim=1)
      y_pred = y_prob.argmax(dim=1)
      correct = (y_pred == labels).sum().item()
      total_correct += correct
      total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

  print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

  return avg_loss, avg_acc

def validate_adversarial(
  config: Config,
  model: nn.Module,
  criterion: nn.Module,
  dataloader: DataLoader[tuple[torch.Tensor, int]],
):
  model.eval()

  atk = PGD(
    model,
    config,
  )

  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  # one_hots = torch.eye(10, 10, dtype=torch.float32, device=config.device)
  for images, labels in dataloader:
    images = images.to(config.device)
    labels = labels.to(config.device)

    adv_images = atk(images, labels)

    with torch.no_grad():
      y_logit = model(adv_images).squeeze(1)
    loss = criterion(y_logit, labels)
    total_loss += loss.item()

    y_prob = F.softmax(y_logit, dim=1)
    y_pred = y_prob.argmax(dim=1)
    correct = (y_pred == labels).sum().item()
    total_correct += correct
    total_samples += labels.size(0)

  avg_loss = total_loss / len(dataloader)
  avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

  return avg_loss, avg_acc

def run(config: Config):
  transforms = T.Compose([
    # T.ToTensor(),
    T.RandomRotation((-30, 30), T.InterpolationMode.NEAREST)
  ])

  train_dataset = get_dataset(
    config.dataset.name,
    config.input_path,
  )
  train_dataset = Subset(train_dataset, range(config.train.subset_size))

  val_dataset = get_dataset(
    config.dataset.name,
    config.input_path,
    train=False,
  )

  train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.dataset.num_workers,
    generator=g,
    # pin_memory=True if config.dataset.num_workers else False,
    # persistent_workers=True
  )
  val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.val.batch_size,
    shuffle=False,
    num_workers=config.dataset.num_workers,
    generator=g,
    # pin_memory=True if config.dataset.num_workers else False,
    # persistent_workers=True
  )

  model = get_model(config.model, config).to(memory_format=torch.channels_last)
  # model = cast(nn.Module, torch.compile(model, mode='default'))

  optimizer = get_optimizer(config.train.optimizer)(
    model.parameters(),
    lr=config.train.learning_rate,
    weight_decay=config.train.weight_decay,
  )

  loss_fn = get_loss_func(config)

  tqdm_ = tqdm(range(config.train.num_epochs))
  for epoch in tqdm_:
    data: dict[str, float] = {}

    train_loss, train_acc = train_one_epoch(
      config,
      model,
      optimizer,
      loss_fn,
      train_dataloader,
    )

    if config.adversarial.compute_robust:
      adv_loss, adv_acc = validate_adversarial(
        config,
        model,
        loss_fn,
        val_dataloader
      )

def main():
  with hydra.initialize_config_dir(config_dir=str(CONFIG_PATH), version_base='1.1'):
    yaml_cfg = hydra.compose(config_name=args.config)

    default_cfg = OmegaConf.structured(Config)

    cfg = cast(Config, OmegaConf.merge(default_cfg, yaml_cfg))
    # cfg = cast(Config, hydra.compose(config_name=args.config))

  cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  seed_everything(cfg.seed)

  print(cast(dict[str, Any], OmegaConf.to_object(cfg)))

  cfg.input_path = ROOT / cfg.input_dir

  with open(CONFIG_PATH / 'log_config.yaml') as f:
    log_config = yaml.safe_load(f)

  logging.config.dictConfig(log_config)

  torch.set_float32_matmul_precision('high')
  # torch.set_default_dtype(torch.float32)
  run(cfg)

if __name__ == '__main__':
  main()
