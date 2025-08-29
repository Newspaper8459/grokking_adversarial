from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeAlias, overload

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import Dataset
from torchattacks import APGD, CW, FAB, FGSM, PGD, DeepFool
from torchattacks.attack import Attack
from torchvision.models.resnet import ResNet

from utils.dataset import CustomCIFAR10, CustomMNIST
from models.mlp import MLP
from models.preact_resnet import resnet18
from schema.config import Config


@overload
def get_optimizer(name: Literal['adam']) -> type[Adam]:
  ...
@overload
def get_optimizer(name: Literal['adamw']) -> type[AdamW]:
  ...
@overload
def get_optimizer(name: Literal['sgd']) -> type[SGD]:
  ...
@overload
def get_optimizer(name: str) -> type[Adam|AdamW|SGD]:
  ...

def get_optimizer(name: str) -> type[Adam|AdamW|SGD]:
  if name == 'adam':
    optimizer = Adam
  elif name == 'adamw':
    optimizer = AdamW
  elif name == 'sgd':
    optimizer = SGD
  else:
    raise NotImplementedError

  return optimizer

def get_loss_func(config: Config) -> nn.Module:
  if config.train.loss == 'ce':
    loss = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
  elif config.train.loss == 'mse':
    loss = nn.MSELoss()
  else:
    raise NotImplementedError

  return loss

@overload
def _get_batch_norm_func(bn_name: Literal['identity']) -> type[nn.Identity]:
  ...
@overload
def _get_batch_norm_func(bn_name: Literal['bn']) -> type[nn.BatchNorm2d]:
  ...
@overload
def _get_batch_norm_func(bn_name: str) -> type[nn.Module]:
  ...

def _get_batch_norm_func(bn_name: str) -> type[nn.Module]:
  if bn_name == 'identity':
    bn = nn.Identity
  elif bn_name == 'bn':
    bn = nn.BatchNorm2d
  else:
    raise NotImplementedError

  return bn

@overload
def get_model(model_name: Literal['mlp'], config: Config) -> MLP:
  ...
@overload
def get_model(model_name: Literal['resnet18'], config: Config) -> ResNet:
  ...
@overload
def get_model(model_name: str, config: Config) -> nn.Module:
  ...

def get_model(model_name: str, config: Config) -> nn.Module:
  if model_name == 'mlp':
    if config.dataset.name != 'mnist':
      raise NotImplementedError

    model = MLP(
      hidden_dim=config.mlp.hidden_dim,
      hidden_layers=config.mlp.hidden_layers,
    )

  elif model_name == 'preact_resnet18':
    bn = _get_batch_norm_func(config.resnet.bn)
    model = resnet18(config.dataset.num_classes, bn)
  else:
    raise NotImplementedError

  with torch.no_grad():
    for p in model.parameters():
      p.data *= config.model.initialization_scale

  return model.to(config.device)

@overload
def get_attack(attack_name: Literal['apgd'], model: nn.Module) -> APGD:
  ...
@overload
def get_attack(attack_name: Literal['cw'], model: nn.Module) -> CW:
  ...
@overload
def get_attack(attack_name: Literal['deepfool'], model: nn.Module) -> DeepFool:
  ...
@overload
def get_attack(attack_name: Literal['fgsm'], model: nn.Module) -> FGSM:
  ...
@overload
def get_attack(attack_name: Literal['pgd'], model: nn.Module) -> PGD:
  ...
@overload
def get_attack(attack_name: str, model: nn.Module) -> Attack:
  ...

def get_attack(attack_name: str, model: nn.Module) -> Attack:
  if attack_name == 'apgd':
    atk = APGD(
      model,
      norm='Linf',
      eps=16/255,
      steps=20,
      seed=0
    )
  elif attack_name == 'cw':
    atk = CW(
      model
    )
  elif attack_name == 'deepfool':
    atk = DeepFool(
      model
    )
  elif attack_name == 'fgsm':
    atk = FGSM(
      model,
      eps=16/255,
    )
  elif attack_name == 'pgd':
    atk = PGD(
      model,
      eps=16/255,
      alpha=4/255,
      steps=20,
    )
  else:
    raise NotImplementedError

  return atk
