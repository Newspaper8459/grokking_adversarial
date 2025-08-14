from collections.abc import Callable
from typing import List, Type

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet


class PreActBasicBlock(BasicBlock):
  def __init__(
    self,
    inplanes: int,
    planes: int,
    stride: int = 1,
    downsample: nn.Module | None = None,
    groups: int = 1,
    base_width: int = 64,
    dilation: int = 1,
    norm_layer: Callable[..., nn.Module] | None = None
  ) -> None:
    super().__init__(
      inplanes,
      planes,
      stride,
      downsample,
      groups,
      base_width,
      dilation,
      norm_layer
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.bn1(x)
    out = self.relu(out)
    out = self.conv1(out)

    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)

    if self.downsample is not None:
      identity = self.downsample(identity)

    out += identity

    return out

def _preact_resnet(
  block: type[PreActBasicBlock],
  layers: list[int],
  num_classes: int,
  norm_layer: type[nn.Module],
) -> ResNet:
  model = ResNet(
    block,
    layers=layers,
    num_classes=num_classes,
    norm_layer=norm_layer
  )

  return model

def resnet18(num_classes: int, norm_layer: type[nn.Module]):
  return _preact_resnet(PreActBasicBlock, [2, 2, 2, 2], num_classes, norm_layer)
