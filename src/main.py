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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from schema.config import Config
from utils.attacks import PGD
from utils.constants import (
  get_loss_func,
  get_model,
  get_optimizer,
)
from utils.dataset import get_dataloader
from workflow import train, validate

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

g = torch.Generator()
g.manual_seed(0)

def run(config: Config):
  transforms = T.Compose([
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
  ])

  train_dataloader = get_dataloader(
    config,
    transforms=transforms,
    subset_size=config.train.subset_size
  )
  val_dataloader = get_dataloader(
    config,
    train=False,
    transforms=transforms
  )

  model = get_model(config.model.name, config).to(
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

  loss_fn = get_loss_func(config).to(config.device, non_blocking=True)

  # tqdm_ = tqdm(range(config.train.num_epochs), )
  # for epoch in tqdm_:

  if config.validate_only:
    assert config.checkpoint_path is not None
  else:
    train(
      config,
      model,
      optimizer,
      loss_fn,
      train_dataloader
    )

    config.checkpoint_path = config.output_path

  validate(
    config,
    model,
    loss_fn,
    train_dataloader,
    val_dataloader,
  )

  if config.save_model:
    state_dict = model.state_dict()
    torch.save(state_dict, config.output_path / f'{config.train.optimization_steps}.pt')

def main():
  with hydra.initialize_config_dir(config_dir=str(CONFIG_PATH), version_base='1.1'):
    yaml_config = hydra.compose(config_name=args.config)

    default_config = OmegaConf.structured(Config)

    config = cast(Config, OmegaConf.merge(default_config, yaml_config))
    # config = cast(Config, hydra.compose(config_name=args.config))

  config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  seed_everything(config.seed)
  if config.resume.enable and config.validate_only:
    raise AttributeError('{config.resume.enable} and {config.validate_only} cannot be True at the same time.')
  if (config.validate_only or config.resume.enable) and config.checkpoint_dir is None:
    raise AttributeError('You must specify {config.checkpoint_dir} if {config.resume.enable} or {config.validate_only} is set to True.')

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
  if config.checkpoint_dir is not None:
    config.checkpoint_path = ROOT / config.checkpoint_dir

  if not config.log_steps:
    # config.log_steps = list(range(100)) + list(range(100, config.train.optimization_steps, 10))
    config.log_steps = np.unique(np.clip(
      np.logspace(0, np.log10(config.train.optimization_steps), 5000).astype(int),
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

  run(config)

if __name__ == '__main__':
  main()
