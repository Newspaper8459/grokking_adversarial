from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent

transforms = T.Compose([
  T.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(
  ROOT / 'input',
  transform=transforms,
  download=True
)
train_dataset = Subset(train_dataset, range(1000))
val_dataset = torchvision.datasets.MNIST(
  ROOT / 'input',
  train=False,
  transform=transforms,
  download=True
)

