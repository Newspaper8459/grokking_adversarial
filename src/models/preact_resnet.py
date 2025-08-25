from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet


# class PreActBasicBlock(BasicBlock):
#   def __init__(
#     self,
#     inplanes: int,
#     planes: int,
#     stride: int = 1,
#     downsample: nn.Module | None = None,
#     groups: int = 1,
#     base_width: int = 64,
#     dilation: int = 1,
#     norm_layer: Callable[..., nn.Module] | None = None
#   ) -> None:
#     super().__init__(
#       inplanes,
#       planes,
#       stride,
#       downsample,
#       groups,
#       base_width,
#       dilation,
#       norm_layer
#     )

#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     identity = x

#     out = self.bn1(x)
#     out = self.relu(out)
#     out = self.conv1(out)

#     out = self.bn2(out)
#     out = self.relu(out)
#     out = self.conv2(out)

#     if self.downsample is not None:
#       identity = self.downsample(identity)

#     out += identity

#     return out

# def resnet18(
# #   block: type[PreActBasicBlock],
# #   layers: list[int],
#   num_classes: int,
#   norm_layer: type[nn.Module],
# ) -> ResNet:
#   model = ResNet(
#     PreActBasicBlock,
#     layers=[2, 2, 2, 2],
#     num_classes=num_classes,
#     norm_layer=norm_layer,
#   )

#   return model

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, **kwargs):
        super(PreActBlock, self).__init__()
        if bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        else:
            self.bn1 = nn.Identity()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if bn:
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn2 = nn.Identity()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_channels=64, bn=True):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv2d(3, c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1, bn=bn)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2, bn=bn)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2, bn=bn)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2, bn=bn)
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, bn=True):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn=bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet18(*args, **kwargs) -> PreActResNet:
  ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
  model = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=10, init_channels=16, bn=False)
  return model
