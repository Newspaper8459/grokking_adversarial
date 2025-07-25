from pathlib import Path
from typing import Any, Callable

import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST


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
