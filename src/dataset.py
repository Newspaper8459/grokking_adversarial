from array import array
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision.transforms as T
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import (
  IntDecoder,
  RandomResizedCropRGBImageDecoder,
  SimpleRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
  Convert,
  Cutout,
  NormalizeImage,
  RandomHorizontalFlip,
  RandomTranslate,
  Squeeze,
  ToDevice,
  ToTensor,
  ToTorchImage,
)
from ffcv.traversal_order import Random
from ffcv.writer import DatasetWriter
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, MNIST

from schema.config import Config


class CustomMNIST(MNIST):
  def __init__(
    self,
    root: str | Path,
    train: bool = True,
    transform: Callable[..., Any] | None = None,
    target_transform: Callable[..., Any] | None = None,
    download: bool = False
  ) -> None:
    super().__init__(root, train, transform, target_transform, download)

    self.data = self.data.float() / 255.0
    self.data = self.data.unsqueeze(1)
    if self.transform is not None:
      self.data = self.transform(self.data)
    self.data_: list[torch.Tensor] = list(self.data)
    self.targets: list[int] = self.targets.tolist()

  def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
    img, target = self.data_[index], self.targets[index]

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img.float(), target

class CustomCIFAR10(CIFAR10):
  def __init__(
    self,
    root: str | Path,
    train: bool = True,
    transform: Callable[..., Any] | None = None,
    target_transform: Callable[..., Any] | None = None,
    download: bool = False
  ) -> None:
    super().__init__(root, train, transform, target_transform, download)

    self.data_tensor = torch.tensor(self.data, dtype=torch.float32)
    self.data_tensor = self.data_tensor.permute((0, 3, 1, 2))
    # if self.transform is not None:
      # self.data_tensor = self.transform(self.data)
    self.data_: list[torch.Tensor] = list(self.data_tensor)
    self.targets_: Iterable[int] = array('I', self.targets)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
    img, target = self.data_[idx], self.targets[idx]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

@overload
def _get_dataset(
  dataset_name: Literal['mnist'],
  input_path: Path,
  train: bool = True,
  transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
  download: bool = True
) -> CustomMNIST:
  ...
@overload
def _get_dataset(
  dataset_name: Literal['cifar-10'],
  input_path: Path,
  train: bool = True,
  transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
  download: bool = True
) -> CustomCIFAR10:
  ...
@overload
def _get_dataset(
  dataset_name: str,
  input_path: Path,
  train: bool = True,
  transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
  download: bool = True
) -> Dataset[Any]:
  ...

def _get_dataset(
  dataset_name: str,
  input_path: Path,
  train: bool = True,
  transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
  download: bool = True
) -> Dataset[Any]:
  if dataset_name == 'mnist':
    dataset = CustomMNIST(
      input_path,
      train=train,
      transform=transforms,
      download=download
    )
  elif dataset_name == 'cifar-10':
    # transforms = T.Compose([
    #   T.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    # ])
    # dataset = CustomCIFAR10(
    #   input_path,
    #   train=train,
    #   transform=transforms,
    #   download=download
    # )
    dataset = CIFAR10(
      input_path,
      train=train,
      transform=transforms,
      download=download,
    )
  else:
    raise NotImplementedError

  return dataset

g = torch.Generator()
g.manual_seed(0)

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

MEAN_DICT: dict[str, npt.NDArray[np.float32]] = {
  'cifar-10': np.array(CIFAR_MEAN, dtype=np.float32),
}

STD_DICT: dict[str, npt.NDArray[np.float32]] = {
  'cifar-10': np.array(CIFAR_STD, dtype=np.float32)
}

def get_dataloader(
  config: Config,
  train: bool = True,
  transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
  download: bool = True,
) -> DataLoader[tuple[torch.Tensor, int]] | Loader:
  if config.dataset.format == 'custom':
    dataset = _get_dataset(
      config.dataset.name,
      input_path=config.input_path,
      train=train,
      transforms=transforms,
      download=download,
    )

    if train:
      dataset = Subset(dataset, range(config.train.subset_size))

    dataloader = DataLoader(
      dataset,
      batch_size=config.train.batch_size if train else config.val.batch_size,
      shuffle=train,
      num_workers=config.dataset.num_workers,
      generator=g,
      pin_memory=bool(config.dataset.num_workers),
      persistent_workers=bool(config.dataset.num_workers),
    )
  elif config.dataset.format == 'ffcv':
    mode = 'train' if train else 'val'

    if config.dataset.name not in ['cifar-10', 'mnist']:
      raise NotImplementedError

    if not (config.input_path / f'{config.dataset.name}_{mode}.ffcv').exists():
      dataset = CIFAR10(root=config.input_path, train=train, download=True)

      ffcv_path = config.input_path / f'{config.dataset.name}_{mode}.ffcv'

      fields = {
        'image': RGBImageField(),
        'label': IntField(),
      }

      writer = DatasetWriter(ffcv_path, fields, num_workers=4)
      writer.from_indexed_dataset(dataset)

    image_pipelines: list[nn.Module | Operation] = [
      SimpleRGBImageDecoder(),
    ]
    if train and False:
      image_pipelines.extend([
        RandomHorizontalFlip(),
        RandomTranslate(padding=2, fill=MEAN_DICT[config.dataset.name]),
        Cutout(4, MEAN_DICT[config.dataset.name]),
      ])

    image_pipelines.extend([
      ToTensor(),
      ToDevice(torch.device(config.device), non_blocking=True),
      ToTorchImage(),
      Convert(torch.float32),
      T.Normalize(
        MEAN_DICT[config.dataset.name],
        STD_DICT[config.dataset.name]
      ),
    ])

    dataloader = Loader(
      str(config.input_path / f'{config.dataset.name}_{mode}.ffcv'),
      batch_size=config.train.batch_size if train else config.val.batch_size,
      num_workers=config.dataset.num_workers,
      os_cache=True,
      order=OrderOption.RANDOM,
      seed=config.seed,
      drop_last=train,
      # indices=list(range(config.train.subset_size)) if train else None,
      pipelines={
        'image': image_pipelines,
        'label': [
          IntDecoder(),
          ToTensor(),
          ToDevice(torch.device(config.device), non_blocking=True),
          Squeeze(),
        ],
      },
    )
  else:
    raise NotImplementedError

  return dataloader
