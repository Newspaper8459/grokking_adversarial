import logging
from functools import cmp_to_key
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from autoattack import AutoAttack
from ffcv.loader import Loader
from torch.utils.data import DataLoader
from tqdm import tqdm

from schema.config import Config
from utils.attacks import PGD
from utils.constants import get_attack


def evaluate(
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

    images = images.to(config.device, non_blocking=True)
    labels = labels.to(config.device, non_blocking=True)

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

def evaluate_adversarial(
  config: Config,
  model: nn.Module,
  criterion: nn.Module,
  dataloader: DataLoader[tuple[torch.Tensor, int]] | Loader,
) -> tuple[float, float]:
  model.eval()
  # atk = AutoAttack(
  #   model,
  #   norm=config.adversarial.norm,
  #   eps=config.adversarial.atk_eps,
  #   seed=config.seed,
  #   device=config.device,
  # )
  atk = get_attack(config.adversarial.name, model)

  total_loss = torch.tensor(0.0, device=config.device)
  total_correct = torch.tensor(0, device=config.device)
  total_samples = 0

  for images, labels in dataloader:
    images = cast(torch.Tensor, images)
    labels = cast(torch.Tensor, labels)

    images = images.to(config.device, non_blocking=True).contiguous()
    labels = labels.to(config.device, non_blocking=True).contiguous()

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

def validate(
  config: Config,
  model: nn.Module,
  criterion: nn.Module,
  train_dataloader: DataLoader[tuple[torch.Tensor, int]] | Loader,
  val_dataloader: DataLoader[tuple[torch.Tensor, int]] | Loader,
):
  logger = logging.getLogger('__main__')
  assert config.checkpoint_path is not None

  checkpoints = list(config.checkpoint_path.glob('*.pt'))
  checkpoints.remove(config.checkpoint_path / 'last.pt')

  def cmp(x: Path, y: Path) -> int:
    s1 = int(x.stem)
    s2 = int(y.stem)

    if s1 < s2:
      return -1
    elif s1 > s2:
      return 1
    else:
      return 0

  checkpoints.sort(key=cmp_to_key(cmp))

  # for step in tqdm(config.log_steps):
  for checkpoint in tqdm(checkpoints):
    stats: dict[str, float] = {}

    if not checkpoint.exists():
      logger.warning(f'Skipped {checkpoint.name}: checkpoint {checkpoint} does not exist.')
      continue

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    model.eval()
    train_loss, train_acc = evaluate(
      config,
      model,
      criterion,
      train_dataloader
    )
    stats['train_loss'] = train_loss
    stats['train_acc'] = train_acc

    val_loss, val_acc = evaluate(
      config,
      model,
      criterion,
      val_dataloader,
    )
    stats['val_loss'] = val_loss
    stats['val_acc'] = val_acc

    if config.adversarial.compute_robust:
      adv_loss, adv_acc = evaluate_adversarial(
        config,
        model,
        criterion,
        val_dataloader
      )
      stats['adv_loss'] = adv_loss
      stats['adv_acc'] = adv_acc

    with torch.no_grad():
      norm = torch.sqrt(
        sum((p.data.float()**2).sum() for p in model.parameters())
      ).item()
      stats['norm'] = norm

    wandb.log(stats, step=int(checkpoint.stem))
