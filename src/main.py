import argparse
import logging
import logging.config
import os
import random
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
import yaml
from ffcv.loader import Loader
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import PGD
from dataset import get_dataloader
from models.preact_resnet import resnet18
from schema.config import Config
from utils.constants import (
  get_loss_func,
  get_model,
  get_optimizer,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs'

parser = argparse.ArgumentParser()

parser.add_argument('--config', '-c', help='config file', default='main.yaml')
args = parser.parse_args()

def fraction_resolver(s: str) -> float:
  f = Fraction(s)
  return float(f)

OmegaConf.register_new_resolver('fraction', fraction_resolver)


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

class EarlyStopping:
  def __init__(self, patience: int=10, verbose: int=0):
    self.cnt = 0
    self.best_loss = float('inf')
    self.patience = patience
    self.verbose = verbose

  def __call__(self, loss: float) -> Any:
    if self.best_loss < loss:
      self.cnt += 1

      if self.cnt >= self.patience:
        if self.verbose:
          print('early stopping')
        return True
    else:
      self.cnt = 0
      self.best_loss = loss

    return False

def test(
  config: Config,
  model: nn.Module,
  criterion: nn.Module,
  dataloader: DataLoader[tuple[torch.Tensor, int]] | Loader,
) -> tuple[float, float]:
  model.eval()

  total_loss = torch.tensor(0.0, device=config.device)
  total_correct = torch.tensor(0, device=config.device)
  total_samples = 0

  for images, labels in dataloader:
    images = cast(torch.Tensor, images)
    labels = cast(torch.Tensor, labels)

    images = images.cuda()
    labels = labels.cuda()

    with torch.no_grad():
      y_logit = model(images)
      loss = criterion(y_logit, labels)
    total_loss += loss

    y_prob = F.softmax(y_logit, dim=1)
    y_pred = y_prob.argmax(dim=1)
    correct = (y_pred == labels).sum()
    total_correct += correct
    total_samples += labels.shape[0]

  avg_loss = total_loss / len(dataloader)
  avg_acc = total_correct / total_samples if total_samples > 0 else torch.tensor(0.0)

  return avg_loss.item(), avg_acc.item()

def validate_adversarial(
  config: Config,
  model: nn.Module,
  criterion: nn.Module,
  dataloader: DataLoader[tuple[torch.Tensor, int]] | Loader,
) -> tuple[float, float]:
  model.eval()
  atk = PGD(
    model,
    config=config,
  )
  atk = PGD(
    model,
    config,
  )

  total_loss = torch.tensor(0.0, device=config.device)
  total_correct = torch.tensor(0, device=config.device)
  total_samples = 0

  for images, labels in dataloader:
    images = cast(torch.Tensor, images)
    labels = cast(torch.Tensor, labels)

    images = images.cuda()
    labels = labels.cuda()

    adv_images = atk(images, labels)

    with torch.no_grad():
      y_logit = model(adv_images)
      loss = criterion(y_logit, labels)
    total_loss += loss

    y_prob = F.softmax(y_logit, dim=1)
    y_pred = y_prob.argmax(dim=1)
    correct = (y_pred == labels).sum()
    total_correct += correct
    total_samples += labels.shape[0]

  avg_loss = total_loss / len(dataloader)
  avg_acc = total_correct / total_samples if total_samples > 0 else torch.tensor(0.0)

  return avg_loss.item(), avg_acc.item()

def run(config: Config):
  transforms = T.Compose([
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
  ])

  train_dataloader = get_dataloader(config, transforms=transforms)
  val_dataloader = get_dataloader(config, train=False, transforms=transforms)

  model = get_model(config.model, config).to(
    device=config.device,
    non_blocking=True,
    memory_format=torch.channels_last
  )
  model = cast(nn.Module, model)
  # model = cast(nn.Module, torch.compile(model, mode='reduce-overhead'))

  optimizer = get_optimizer(config.train.optimizer)(
    model.parameters(),
    lr=config.train.learning_rate,
    weight_decay=config.train.weight_decay,
  )

  loss_fn = get_loss_func(config).cuda()

  if config.early_stopping.enable:
    early_stopping = EarlyStopping(
      config.early_stopping.patience,
      config.early_stopping.verbose,
    )

  # tqdm_ = tqdm(range(config.train.num_epochs), )
  # for epoch in tqdm_:
  steps = 0
  log_steps = set(config.log_steps)
  with tqdm(total=len(log_steps)) as pbar:
    data: dict[str, float] = {}
    data_str: dict[str, str] = {}
    data_str['steps'] = '0'
    while steps < config.train.optimization_steps:
      for images, labels in train_dataloader:
        model.train()
        optimizer.zero_grad()

        if steps >= config.train.optimization_steps:
          break

        images = cast(torch.Tensor, images)
        labels = cast(torch.Tensor, labels)

        images = images.cuda()
        labels = labels.cuda()

        y_logit = model(images)
        loss = loss_fn(y_logit, labels)

        loss.backward()
        optimizer.step()

        if steps in log_steps:
          train_loss, train_acc = test(
            config,
            model,
            loss_fn,
            train_dataloader,
          )
          data['train_loss_curve'] = train_loss
          data['train_acc'] = train_acc

          val_loss, val_acc = test(
            config,
            model,
            loss_fn,
            val_dataloader
          )
          data['val_loss_curve'] = val_loss
          data['val_acc'] = val_acc

          if config.adversarial.compute_robust:
            adv_loss, adv_acc = validate_adversarial(
              config,
              model,
              loss_fn,
              val_dataloader
            )
            data['adv_loss_curve'] = adv_loss
            data['adv_acc'] = adv_acc

          with torch.no_grad():
            norm = torch.sqrt(
              sum((p.data.float()**2).sum() for p in model.parameters())
            ).item()
            data['norm'] = norm

          data['steps'] = steps

          for k, v in data.items():
            if k == 'steps':
              data_str[k] = str(v)
            else:
              data_str[k] = f'{v:.4f}'

          wandb.log(data, step=steps)
          pbar.update(1)

          if config.early_stopping.enable:
            if early_stopping(val_loss):
              return True

        data_str['steps'] = str(steps)
        tqdm.set_postfix(pbar, data_str)
        steps += 1

  if config.save_model:
    state_dict = model.state_dict()
    torch.save(state_dict, config.output_path)

  return False

def main():
  with hydra.initialize_config_dir(config_dir=str(CONFIG_PATH), version_base='1.1'):
    yaml_config = hydra.compose(config_name=args.config)

    default_config = OmegaConf.structured(Config)

    config = cast(Config, OmegaConf.merge(default_config, yaml_config))
    # config = cast(Config, hydra.compose(config_name=args.config))

  config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # seed_everything(config.seed)

  if config.debug_mode:
    mode = 'offline'
  else:
    mode = 'online'
  wandb.init(
    project=config.wandb.project,
    mode=mode,
    dir=ROOT / config.output_dir,
    config=cast(dict[str, Any], OmegaConf.to_object(config))
  )

  config.input_path = ROOT / config.input_dir
  config.output_path = ROOT / config.output_dir / wandb.run.dir

  if not config.log_steps:
    # config.log_steps = list(range(100)) + list(range(100, config.train.optimization_steps, 10))
    config.log_steps = np.unique(np.clip(
      np.logspace(0, np.log10(config.train.optimization_steps), 500).astype(int),
      0,
      config.train.optimization_steps,
    )).tolist()

  with open(CONFIG_PATH / 'log_config.yaml') as f:
    log_config = yaml.safe_load(f)

  logging.config.dictConfig(log_config)
  file_handler = logging.FileHandler(config.output_path / 'out.log')
  file_handler.setLevel('INFO')
  file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"))

  for name in ['__main__', 'same_hierarchy', 'lower.sub']:
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)

  # torch.set_float32_matmul_precision('highest')
  # torch.set_default_dtype(torch.float32)
  print(cast(dict[str, Any], OmegaConf.to_object(config)))

  early_stopped = run(config)
  if early_stopped:
    run_adv(config)


if __name__ == '__main__':
  main()
