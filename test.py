from itertools import islice
from pathlib import Path
from time import time

import autoattack
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchattacks import APGD, CW, FAB, FGSM, PGD, DeepFool
from torchvision.datasets import CIFAR10, MNIST, ImageNet
from torchvision.models.resnet import resnet18
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
p = ROOT / 'checkpoints/baseline'
a = list(p.glob('d+.pt'))
print(a)

APGD(
  model=None,
  norm=None,
  eps=None,
  steps=None,
  seed=None
)

CW(
  model
)

DeepFool(
  model
)

