import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(
  in_planes: int,
  out_planes: int,
  stride: int = 1,
  groups: int = 1,
  dilation: int = 1
) -> nn.Conv2d:
  """3x3 convolution with padding"""
  return nn.Conv2d(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=stride,
    padding=dilation,
    groups=groups,
    bias=False,
    dilation=dilation,
  )

class PreActBasicBlock(nn.Module):
  expansion: int = 1

  def __init__(
    self,
    inplanes: int,
    planes: int,
    stride: int = 1,
    norm_layer: type[nn.Module] | None = None,
  ) -> None:
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.stride = stride

    if stride != 1 or inplanes != self.expansion*planes:
      self.shortcut = nn.Conv2d(
        inplanes,
        self.expansion*planes,
        kernel_size=1,
        stride=stride,
        bias=False,
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    out = self.bn1(x)
    out = self.relu(out)

    shortcut = out

    out = self.conv1(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)

    shortcut = self.shortcut(shortcut) if hasattr(self, 'shortcut') else x

    out += shortcut

    return out

class PreActResNet(nn.Module):
  def __init__(
    self,
    block: type[PreActBasicBlock],
    layers: list[int],
    num_classes: int = 1000,
    init_channels: int = 64,
    norm_layer: type[nn.Module] | None = None,
    imagenet_stem: bool = True,
  ) -> None:
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d

    self.in_planes = init_channels
    c = init_channels

    if imagenet_stem:
      self.conv1 = nn.Conv2d(
        3,
        c,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
      )
      self.bn1 = norm_layer(c)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    else:
      self.conv1 = nn.Conv2d(
        3,
        c,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
      )
      self.bn1 = None
      self.relu = None
      self.maxpool = None

    self.layer1 = self._make_layer(
      block=block,
      planes=c,
      blocks=layers[0],
      norm_layer=norm_layer,
      stride=1
    )
    self.layer2 = self._make_layer(
      block=block,
      planes=2*c,
      blocks=layers[1],
      norm_layer=norm_layer,
      stride=2
    )
    self.layer3 = self._make_layer(
      block=block,
      planes=4*c,
      blocks=layers[2],
      norm_layer=norm_layer,
      stride=2
    )
    self.layer4 = self._make_layer(
      block=block,
      planes=8*c,
      blocks=layers[3],
      norm_layer=norm_layer,
      stride=2
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # self.avgpool = nn.AvgPool2d(4)
    self.fc = nn.Linear(8 * c * block.expansion, num_classes)

  def _make_layer(
    self,
    block: type[PreActBasicBlock],
    planes: int,
    blocks: int,
    norm_layer: type[nn.Module],
    stride: int = 1,
  ) -> nn.Sequential:
    strides = [stride] + [1]*(blocks-1)
    layers: list[PreActBasicBlock] = []
    for stride in strides:
      layers.append(
        block(
          self.in_planes,
          planes,
          stride,
          norm_layer
        )
      )
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv1(x)
    if self.bn1 is not None:
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x

def resnet18(
#   layers: list[int],
  num_classes: int,
  norm_layer: type[nn.Module],
) -> PreActResNet:
  model = PreActResNet(
    PreActBasicBlock,
    layers=[2, 2, 2, 2],
    num_classes=num_classes,
    norm_layer=norm_layer,
    init_channels=16,
    imagenet_stem=False
  )

  return model
