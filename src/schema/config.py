from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


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
  atk_epochs: int
  dmax: float
  dmin: float
  atk_eps: float
  atk_alpha: float

@dataclass
class EarlyStoppingConfig:  # noqa: D101
  enable: bool
  patience: int
  verbose: int

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
  save_model: bool
  wandb: WandbConfig
  train: TrainConfig
  val: ValConfig
  local_complexity: LocalComplexityConfig
  adversarial: AdversarialConfig
  early_stopping: EarlyStoppingConfig
  dataset: DatasetConfig
  mnist: MNISTConfig
  model: str
  mlp: MLPConfig
  resnet: ResNetConfig
  log_steps: list[int] = field(default_factory=list)
  device: str = ''
  input_path: Path = Path()
  output_path: Path = Path()
