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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import CustomMNIST
from models.mlp import MLP
from schema.config import Config

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
  torch.backends.cudnn.benchmark = False
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
  optimizer: torch.optim.Optimizer,
  criterion: nn.Module,
  dataloader: Any,
):
  model.train()

  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  one_hots = torch.eye(10, 10, dtype=torch.float32, device=config.device)

  for images, labels in dataloader:
    optimizer.zero_grad()

    images = images.to(config.device)
    labels = labels.to(config.device)
    # with torch.autocast(config.device, dtype=torch.float16):
    y_logit = model(images).squeeze(1)
    loss = criterion(one_hots[labels], y_logit)
    total_loss += loss.item()

    loss.backward()
    optimizer.step()

    # scaler.scale(loss).backward()
    # scaler.unscale_(optimizer)
    # nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_norm)
    # scaler.step(optimizer)
    # scaler.update()

    y_prob = F.softmax(y_logit, dim=1)
    y_pred = y_prob.argmax(dim=1)
    correct = (y_pred == labels).sum().item()
    total_correct += correct
    total_samples += labels.size(0)

  avg_loss = total_loss / len(dataloader)
  avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

  return avg_loss, avg_acc

def validate(
  config: Config,
  model: nn.Module,
  criterion: nn.Module,
  dataloader: Any,
):
  model.eval()

  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  one_hots = torch.eye(10, 10, dtype=torch.float32, device=config.device)
  for images, labels in dataloader:
    images = images.to(config.device)
    labels = labels.to(config.device)

    with torch.no_grad():
      y_logit = model(images).squeeze(1)
    loss = criterion(one_hots[labels], y_logit)
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

  train_dataset = CustomMNIST(
    config.input_path,
    # transform=transforms,
    download=True
  )
  train_dataset = Subset(train_dataset, range(config.train.subset_size))
  val_dataset = CustomMNIST(
    config.input_path,
    train=False,
    # transform=transforms,
    download=True
  )

  train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    generator=g,
    pin_memory=True if config.num_workers else False,
    persistent_workers=True
  )
  val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.val.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    generator=g,
    pin_memory=True if config.num_workers else False,
    persistent_workers=True
  )

  model = MLP(config.mlp.hidden_dim, config.mlp.hidden_layers).to(config.device)
  model = cast(MLP, torch.compile(model, mode='default'))

  with torch.no_grad():
    for p in model.parameters():
      p.data *= config.train.initialization_scale

  optimizer = AdamW(
    model.parameters(),
    lr=config.train.learning_rate,
    # betas=(0.9, 0.98),
    weight_decay=config.train.weight_decay,
  )

  loss_fn = nn.MSELoss()

  logger = logging.getLogger('__main__')

  tqdm_ = tqdm(range(config.train.num_epochs))
  for epoch in tqdm_:
    train_loss, train_acc = train_one_epoch(
      config,
      model,
      optimizer,
      loss_fn,
      train_dataloader,
    )

    val_loss, val_acc = validate(
      config,
      model,
      loss_fn,
      val_dataloader
    )

    tqdm.set_postfix(
      tqdm_,
      {
        "train loss": f'{train_loss:.4f}',
        "train acc": f'{train_acc:.4f}',
        "val loss": f'{val_loss:.4f}',
        "val acc": f'{val_acc:.4f}'
      }
    )

    wandb.log({
      'train_loss_curve': train_loss,
      'train_acc': train_acc,
      'val_loss_curve': val_loss,
      'val_acc': val_acc,
    }, step=epoch)



def main():
  ###################################################
  # TODO: float64統一
  ###################################################
  with hydra.initialize_config_dir(config_dir=str(CONFIG_PATH), version_base='1.1'):
    cfg = cast(Config, hydra.compose(config_name=args.config))

  cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  seed_everything(cfg.seed)

  print(cast(dict[str, Any], OmegaConf.to_object(cfg)))
  if cfg.debug_mode:
    mode = 'offline'
  else:
    mode = 'online'
  wandb.init(
    project=cfg.wandb.project,
    mode=mode,
    dir=ROOT / cfg.output_dir,
    config=cast(dict[str, Any], OmegaConf.to_object(cfg))
  )

  cfg.input_path = ROOT / cfg.input_dir
  cfg.output_path = ROOT / cfg.output_dir / wandb.run.dir

  with open(CONFIG_PATH / 'log_config.yaml') as f:
    log_config = yaml.safe_load(f)

  logging.config.dictConfig(log_config)
  file_handler = logging.FileHandler(cfg.output_path / 'out.log')
  file_handler.setLevel('INFO')
  file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"))

  for name in ['__main__', 'same_hierarchy', 'lower.sub']:
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)

  # torch.set_float32_matmul_precision('high')
  # torch.set_default_dtype(torch.float32)
  run(cfg)

if __name__ == '__main__':
  main()
