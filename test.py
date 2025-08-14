from itertools import islice
from pathlib import Path
from time import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10, ImageNet
from torchvision.models.resnet import resnet18
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent

a = np.unique(np.clip(np.logspace(0, np.log10(500000), 1000), 0, 500000).astype(int))

print(a)
