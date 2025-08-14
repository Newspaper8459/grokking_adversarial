from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig


@dataclass
class WandbConfig:  # noqa: D101
  project: str

class LossType(str, Enum):  # noqa: D101
  ce = "ce"
  mse = "mse"

@dataclass
class TrainConfig:  # noqa: D101
  optimization_steps: int
  batch_size: int
  optimizer: str
  learning_rate: float
  weight_decay: float
  loss: LossType
  label_smoothing: float
  subset_size: int
  grad_norm: float

@dataclass
class ValConfig:  # noqa: D101
  batch_size: int

@dataclass
class LocalComplexityConfig:  # noqa: D101
  compute_lc: bool
  approx_n: int
  n_frame: int
  r_frame: float
  lc_batch_size: int

@dataclass
class AdversarialConfig:  # noqa: D101
  compute_robust: bool
  atk_eps_numerator: int
  atk_alpha_numerator: int
  atk_epochs: int
  dmax: float
  dmin: float
  atk_eps: float = .0
  atk_alpha: float = .0

@dataclass
class DatasetConfig:  # noqa: D101
  name: str
  format: str
  num_classes: int
  num_workers: int

@dataclass
class MNISTConfig:  #noqa: D101
  initialization_scale: int

@dataclass
class MLPConfig:  # noqa: D101
  hidden_dim: int
  hidden_layers: int

@dataclass
class ResNetConfig:  # noqa: D101
  bn: str

@dataclass
class Config:  # noqa: D101
  debug_mode: bool
  seed: int
  input_dir: str
  output_dir: str
  use_amp: bool
  wandb: WandbConfig
  train: TrainConfig
  val: ValConfig
  local_complexity: LocalComplexityConfig
  adversarial: AdversarialConfig
  dataset: DatasetConfig
  mnist: MNISTConfig
  model: str
  mlp: MLPConfig
  resnet: ResNetConfig
  log_steps: list[int] = field(default_factory=list)
  device: str = ''
  input_path: Path = Path()
  output_path: Path = Path()
