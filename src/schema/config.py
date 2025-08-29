from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass
class ResumeConfig:  # noqa: D101
  enable: bool
  step: int | None = None

@dataclass
class WandbConfig:  # noqa: D101
  project: str

class LossType(str, Enum):  # noqa: D101
  ce = "ce"
  mse = "mse"

class NormType(str, Enum):  # noqa: D101
  L1 = "L1"
  L2 = "L2"
  Linf = "Linf"

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
  inc_centroid: bool

@dataclass
class AdversarialConfig:  # noqa: D101
  compute_robust: bool
  name: str
  atk_eps: float
  atk_alpha: float
  norm: NormType
  # atk_epochs: int
  # dmax: float
  # dmin: float

@dataclass
class TrainAdvSamplesConfig:  # noqa: D101
  enable: bool
  # patience: int
  # verbose: int
  interrupt_step: int
  train_size: int

@dataclass
class DatasetConfig:  # noqa: D101
  name: str
  format: str
  num_classes: int
  num_workers: int

@dataclass
class ModelConfig:  # noqa: D101
  name: str
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
  validate_only: bool
  save_model: bool
  resume: ResumeConfig
  wandb: WandbConfig
  train: TrainConfig
  val: ValConfig
  local_complexity: LocalComplexityConfig
  adversarial: AdversarialConfig
  train_adv_samples: TrainAdvSamplesConfig
  dataset: DatasetConfig
  model: ModelConfig
  mlp: MLPConfig
  resnet: ResNetConfig
  log_steps: list[int] = field(default_factory=list)
  device: str = ''
  input_path: Path = Path()
  output_path: Path = Path()
  checkpoint_dir: str | None = None
  checkpoint_path: Path | None = None
